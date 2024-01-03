//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "memory.h"
#include "sentencepiece/sentencepiece_processor.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

static constexpr int NUM_LAYERS = 32;
static constexpr int MAX_LEN = 512;
static constexpr float ATTENTION_MASK = -10000.;

struct LLama2Inner {
  virtual int forward_first(const std::vector<int> &) = 0;
  virtual int forward_next(int token) = 0;
  virtual void init(const std::vector<int> &devices,
                    std::string_view model_path) = 0;
  virtual void step_back(const bm_tensor_t &kv,
                         const bm_tensor_t &kv_cache) = 0;
  virtual void deinit() = 0;

  int token_length;
};

struct LLama2Impl : LLama2Inner {
  int device_num;
  bm_handle_t bm_handle;
  std::vector<bm_handle_t> handles;
  void *p_bmrt;
  const bm_net_info_t *net_blocks[NUM_LAYERS];
  const bm_net_info_t *net_blocks_cache[NUM_LAYERS];
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<bm_tensor_t> inputs_embed_512, outputs_embed_512;
  std::vector<bm_tensor_t> inputs_pid, next_pid, inputs_attention,
      next_attention;
  std::vector<bm_tensor_t> past_key[NUM_LAYERS], past_value[NUM_LAYERS];
  std::vector<bm_tensor_t> present_key_cache, present_value_cache;
  std::vector<bm_tensor_t> inputs_lm, outputs_lm;
  std::string name_embed;
  std::string name_embed_cache;
  std::string name_lm;
  std::string name_blocks[NUM_LAYERS];
  std::string name_blocks_cache[NUM_LAYERS];

  int forward_first(const std::vector<int> &) override;
  int forward_next(int token) override;
  void init(const std::vector<int> &devices,
            std::string_view model_path) override;
  void step_back(const bm_tensor_t &kv, const bm_tensor_t &kv_cache) override;
  void deinit() override;
};

struct LLama2Impl2 : LLama2Inner {

  int forward_first(const std::vector<int> &) override;
  int forward_next(int token) override;
  void init(const std::vector<int> &devices,
            std::string_view model_path) override;
  void step_back(const bm_tensor_t &kv, const bm_tensor_t &kv_cache) override;
  void deinit() override;

  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  sentencepiece::SentencePieceProcessor sentencepiece;
  const bm_net_info_t *net_blocks[NUM_LAYERS];
  const bm_net_info_t *net_blocks_cache[NUM_LAYERS];
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm;
  bm_tensor_t inputs_embed_512, outputs_embed_512;
  bm_tensor_t inputs_lm, outputs_lm;
  bm_tensor_t inputs_pid, next_pid, inputs_attention, next_attention;
  bm_tensor_t past_key[NUM_LAYERS], past_value[NUM_LAYERS];
  bm_tensor_t present_key[NUM_LAYERS], present_value[NUM_LAYERS];
  bm_tensor_t present_key_cache, present_value_cache;
  std::string name_embed;
  std::string name_lm;
  std::string name_blocks[NUM_LAYERS];
  std::string name_blocks_cache[NUM_LAYERS];
  int EOS;
};

void LLama2Impl2::init(const std::vector<int> &devices,
                       std::string_view model_path) {
  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];
  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  int device_num = devices.size();
  // Is this two lines the same?
  // p_bmrt = bmrt_create_ex(handles.data(), handles.size());
  p_bmrt = bmrt_create_ex(handles.data(), device_num);
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.data());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.data());
  assert(true == ret);
  printf("Done!\n");
  // net names
  name_embed = "embedding";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "block_" + std::to_string(i);
    name_blocks_cache[i] = "block_cache_" + std::to_string(i);
  }

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  for (int i = 0; i < NUM_LAYERS; i++) {
    net_blocks[i] = bmrt_get_network_info(p_bmrt, name_blocks[i].c_str());
    net_blocks_cache[i] =
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str());
  }

  // net device mem
  ret = bmrt_tensor(&inputs_embed_512, p_bmrt, net_embed->input_dtypes[0],
                    net_embed->stages[1].input_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&outputs_embed_512, p_bmrt, net_embed->output_dtypes[0],
                    net_embed->stages[1].output_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_pid, p_bmrt, net_blocks[0]->input_dtypes[1],
                    net_blocks[0]->stages[0].input_shapes[1]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_attention, p_bmrt, net_blocks[0]->input_dtypes[2],
                    net_blocks[0]->stages[0].input_shapes[2]);
  assert(true == ret);

  ret = bmrt_tensor(&next_pid, p_bmrt, net_blocks_cache[0]->input_dtypes[1],
                    net_blocks_cache[0]->stages[0].input_shapes[1]);
  assert(true == ret);

  ret =
      bmrt_tensor(&next_attention, p_bmrt, net_blocks_cache[0]->input_dtypes[2],
                  net_blocks_cache[0]->stages[0].input_shapes[2]);
  assert(true == ret);

  for (int i = 0; i < NUM_LAYERS; i++) {
    ret = bmrt_tensor(&past_key[i], p_bmrt, net_blocks[0]->output_dtypes[1],
                      net_blocks[0]->stages[0].output_shapes[1]);
    assert(true == ret);
    ret = bmrt_tensor(&past_value[i], p_bmrt, net_blocks[0]->output_dtypes[2],
                      net_blocks[0]->stages[0].output_shapes[2]);
    assert(true == ret);
    ret = bmrt_tensor(&present_key[i], p_bmrt, net_blocks[0]->output_dtypes[1],
                      net_blocks[0]->stages[0].output_shapes[1]);
    assert(true == ret);
    ret =
        bmrt_tensor(&present_value[i], p_bmrt, net_blocks[0]->output_dtypes[2],
                    net_blocks[0]->stages[0].output_shapes[2]);
    assert(true == ret);
  }
  ret = bmrt_tensor(&present_key_cache, p_bmrt,
                    net_blocks_cache[0]->output_dtypes[1],
                    net_blocks_cache[0]->stages[0].output_shapes[1]);
  assert(true == ret);
  ret = bmrt_tensor(&present_value_cache, p_bmrt,
                    net_blocks_cache[0]->output_dtypes[2],
                    net_blocks_cache[0]->stages[0].output_shapes[2]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_lm, p_bmrt, net_lm->input_dtypes[0],
                    net_lm->stages[0].input_shapes[0]);
  assert(true == ret);
  ret = bmrt_tensor(&outputs_lm, p_bmrt, net_lm->output_dtypes[0],
                    net_lm->stages[0].output_shapes[0]);
  assert(true == ret);
}

void LLama2Impl2::deinit() {
  bm_free_device(bm_handle, inputs_embed_512.device_mem);
  bm_free_device(bm_handle, outputs_embed_512.device_mem);
  bm_free_device(bm_handle, inputs_lm.device_mem);
  bm_free_device(bm_handle, outputs_lm.device_mem);
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
  bm_free_device(bm_handle, present_key_cache.device_mem);
  bm_free_device(bm_handle, present_value_cache.device_mem);
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_free_device(bm_handle, past_key[i].device_mem);
    bm_free_device(bm_handle, past_value[i].device_mem);
    bm_free_device(bm_handle, present_key[i].device_mem);
    bm_free_device(bm_handle, present_value[i].device_mem);
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void LLama2Impl2::step_back(const bm_tensor_t &kv,
                            const bm_tensor_t &kv_cache) {
  if (token_length >= MAX_LEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / MAX_LEN;
  auto real_size = (token_length - 1) * bytes;
  auto offset = (MAX_LEN - token_length + 1) * bytes;
  auto mem = bm_mem_from_device(bm_mem_get_device_addr(kv.device_mem) + offset,
                                real_size);
  auto mem_cache =
      bm_mem_from_device(bm_mem_get_device_addr(kv_cache.device_mem), bytes);
  auto buffer = new uint8_t[real_size];
  auto buffer_cache = new uint8_t[bytes];
  auto dst = new uint8_t[total_size];
  bm_memcpy_d2s(bm_handle, (void *)buffer, mem);
  bm_memcpy_d2s(bm_handle, (void *)buffer_cache, mem_cache);
  // memset(dst, 0, total_size - real_size);
  memcpy(dst + total_size - real_size - bytes, buffer, real_size);
  memcpy(dst + total_size - bytes, buffer_cache, bytes);
  bm_memcpy_s2d(bm_handle, kv.device_mem, (void *)dst);
  delete[] buffer;
  delete[] buffer_cache;
  delete[] dst;
}

int LLama2Impl2::forward_first(const std::vector<int> &tokens) {
  int input_ids[MAX_LEN] = {0}; // start token
  int position_id[MAX_LEN] = {0};
  float attention_mask[MAX_LEN * MAX_LEN] = {0};
  token_length = tokens.size();

  std::copy(tokens.begin(), tokens.end(), input_ids);
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }

  for (int i = 0; i < MAX_LEN; i++) {
    for (int j = 0; j < MAX_LEN; j++) {
      if (j <= i && i < token_length) {
      } else {
        attention_mask[i * MAX_LEN + j] = ATTENTION_MASK;
      }
    }
  }

  // forward embeding
  bm_memcpy_s2d(bm_handle, inputs_embed_512.device_mem, (void *)input_ids);
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &inputs_embed_512, 1,
                            &outputs_embed_512, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void *)position_id);
  bm_memcpy_s2d(bm_handle, inputs_attention.device_mem, (void *)attention_mask);
  auto inputs_embed = outputs_embed_512;
  inputs_embed.shape = net_blocks[0]->stages[0].input_shapes[0];
  bm_tensor_t inputs_block[3] = {inputs_embed, inputs_pid, inputs_attention};
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(), inputs_block, 3,
                                outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }
  int bytes = inputs_embed.device_mem.size / MAX_LEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm.device_mem, 0,
                     inputs_embed.device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}

int LLama2Impl2::forward_next(int cur_token) {
  float attention_mask[MAX_LEN + 1] = {0};
  for (int i = token_length - 1; i < MAX_LEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;
  // embedding
  outputs_lm.shape = net_embed->stages[0].input_shapes[0];
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &outputs_lm, 1,
                                   &inputs_lm, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // blocks
  bm_memcpy_s2d(bm_handle, next_attention.device_mem, (void *)attention_mask);
  bm_memcpy_s2d(bm_handle, next_pid.device_mem, (void *)&position_id);
  auto inputs_embed = inputs_lm;
  inputs_embed.shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  int bytes = bm_mem_get_device_size(present_key_cache.device_mem);
  int token_offset = (token_length - 1) * bytes;
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t inputs_block[5] = {inputs_embed, next_pid, next_attention,
                                   past_key[i], past_value[i]};
    bm_tensor_t outputs_block[3] = {inputs_embed, present_key_cache,
                                    present_value_cache};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block, 5, outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    bm_memcpy_d2d_byte(bm_handle, past_key[i].device_mem, token_offset,
                       present_key_cache.device_mem, 0, bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[i].device_mem, token_offset,
                       present_value_cache.device_mem, 0, bytes);
  }
  outputs_lm.shape = net_lm->stages[0].output_shapes[0];
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}

static sentencepiece::SentencePieceProcessor tokenizer;

struct LLama2TPU {
  LLama2TPU(LLama2Inner *inner) : inner(inner) { eos = tokenizer.eos_id(); }
  LLama2Inner *inner;
  std::string history{""};
  int round{0};
  int eos;
  ~LLama2TPU() { inner->deinit(); }
  void answer(std::string_view);
  void chat();
};

#define CALL_WITH_ASSERT(X)                                                    \
  do {                                                                         \
    if (!(X)) {                                                                \
      fprintf(stderr,                                                          \
              "Assertion '%s' failed in %s "                                   \
              "at %s:%d\n",                                                    \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__);                    \
      abort();                                                                 \
    }                                                                          \
  } while (false)

template <typename T>
static LLama2TPU from_pretrained(T &&dev_ids, std::string_view model_path) {
  std::string tokenizer_path{"tokenizer.model"};
  size_t last_slash_pos = model_path.rfind('/');
  if (last_slash_pos != std::string::npos)
    tokenizer_path =
        std::string{model_path.substr(0, last_slash_pos + 1)} + tokenizer_path;
  LLama2Inner *inner{nullptr};

  if (dev_ids.size() > 1)
    inner = new LLama2Impl;
  else
    inner = new LLama2Impl2;

  inner->init(dev_ids, model_path);

  if (auto status = tokenizer.Load(tokenizer_path); !status.ok()) {
    std::cerr << status.error_message() << '\n';
    exit(0);
  }
  return LLama2TPU{inner};
}

void dump_tensor(bm_handle_t bm_handle, bm_tensor_t &tensor) {
  auto shape = tensor.shape;
  int size = 1;
  for (int i = 0; i < shape.num_dims; ++i) {
    size *= shape.dims[i];
  }
  std::vector<float> data(size);
  bm_memcpy_d2s(bm_handle, data.data(), tensor.device_mem);
  // std::cout<< data[0] << "\t" << data[data.size()-1] << std::endl;
  auto ptr = data.data();
  ptr[0] = ptr[0];
}

void LLama2Impl::init(const std::vector<int> &devices, std::string_view model) {
  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  device_num = devices.size();
  for (auto d : devices) {
    bm_handle_t h;
    auto ret = bm_dev_request(&h, d);
    assert(ret == BM_SUCCESS);
    handles.push_back(h);
  }
  bm_handle = handles[0];
  // create bmruntime
  p_bmrt = bmrt_create_ex(handles.data(), device_num);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model.data());
  bool ret = bmrt_load_bmodel(p_bmrt, model.data());
  assert(true == ret);
  printf("Done!\n");
  // net names
  name_embed = "embedding";
  name_embed_cache = "embedding_cache";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "block_" + std::to_string(i);
    name_blocks_cache[i] = "block_cache_" + std::to_string(i);
  }

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_embed_cache = bmrt_get_network_info(p_bmrt, name_embed_cache.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  for (int i = 0; i < NUM_LAYERS; i++) {
    net_blocks[i] = bmrt_get_network_info(p_bmrt, name_blocks[i].c_str());
    net_blocks_cache[i] =
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str());
  }

  // net device mem
  inputs_embed_512.resize(net_embed->input_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(
        &inputs_embed_512[i], p_bmrt, net_embed->input_loc_devices[i],
        net_embed->input_dtypes[i], net_embed->stages[0].input_shapes[i]);
    assert(true == ret);
  }

  outputs_embed_512.resize(net_embed->output_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(
        &outputs_embed_512[i], p_bmrt, net_embed->output_loc_devices[i],
        net_embed->output_dtypes[i], net_embed->stages[0].output_shapes[i]);
    assert(ret == true);
  }

  inputs_pid.resize(device_num);
  inputs_attention.resize(device_num);
  int in_num = net_blocks[0]->input_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_pid[i], p_bmrt,
                         net_blocks[0]->input_loc_devices[1 + i * in_num],
                         net_blocks[0]->input_dtypes[1 + i * in_num],
                         net_blocks[0]->stages[0].input_shapes[1 + i * in_num]);
    assert(true == ret);

    ret = bmrt_tensor_ex(&inputs_attention[i], p_bmrt,
                         net_blocks[0]->input_loc_devices[2 + i * in_num],
                         net_blocks[0]->input_dtypes[2 + i * in_num],
                         net_blocks[0]->stages[0].input_shapes[2 + i * in_num]);
    assert(true == ret);
  }

  next_pid.resize(device_num);
  next_attention.resize(device_num);
  int in_num_cache = net_blocks_cache[0]->input_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(
        &next_pid[i], p_bmrt,
        net_blocks_cache[0]->input_loc_devices[1 + i * in_num_cache],
        net_blocks_cache[0]->input_dtypes[1 + i * in_num_cache],
        net_blocks_cache[0]->stages[0].input_shapes[1 + i * in_num_cache]);
    assert(true == ret);

    ret = bmrt_tensor_ex(
        &next_attention[i], p_bmrt,
        net_blocks_cache[0]->input_loc_devices[2 + i * in_num_cache],
        net_blocks_cache[0]->input_dtypes[2 + i * in_num_cache],
        net_blocks_cache[0]->stages[0].input_shapes[2 + i * in_num_cache]);
    assert(true == ret);
  }

  int out_num = net_blocks[0]->output_num / device_num;
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i].resize(device_num);
    past_value[i].resize(device_num);
    for (int j = 0; j < device_num; j++) {
      ret = bmrt_tensor_ex(
          &past_key[i][j], p_bmrt,
          net_blocks[0]->output_loc_devices[1 + j * out_num],
          net_blocks[0]->output_dtypes[1 + j * out_num],
          net_blocks[0]->stages[0].output_shapes[1 + j * out_num]);
      assert(true == ret);
      ret = bmrt_tensor_ex(
          &past_value[i][j], p_bmrt,
          net_blocks[0]->output_loc_devices[2 + j * out_num],
          net_blocks[0]->output_dtypes[2 + j * out_num],
          net_blocks[0]->stages[0].output_shapes[2 + j * out_num]);
      assert(true == ret);
    }
  }

  present_key_cache.resize(device_num);
  present_value_cache.resize(device_num);
  inputs_lm.resize(device_num);
  outputs_lm.resize(device_num);
  for (int i = 0; i < device_num; ++i) {
    present_key_cache[i] = past_key[0][i];
    present_value_cache[i] = past_value[0][i];
    present_key_cache[i].shape.dims[1] = 1;
    present_value_cache[i].shape.dims[1] = 1;

    ret = bmrt_tensor_ex(&inputs_lm[i], p_bmrt, i, net_lm->input_dtypes[0],
                         net_lm->stages[0].input_shapes[0]);
    assert(true == ret);
    ret = bmrt_tensor_ex(&outputs_lm[i], p_bmrt, i, net_lm->output_dtypes[0],
                         net_lm->stages[0].output_shapes[0]);
    assert(true == ret);
  }
}

void LLama2Impl::deinit() {
  for (int i = 0; i < device_num; ++i) {
    bm_free_device(handles[i], inputs_embed_512[i].device_mem);
    bm_free_device(handles[i], outputs_embed_512[i].device_mem);
    bm_free_device(handles[i], inputs_pid[i].device_mem);
    bm_free_device(handles[i], next_pid[i].device_mem);
    bm_free_device(handles[i], inputs_attention[i].device_mem);
    bm_free_device(handles[i], next_attention[i].device_mem);
    bm_free_device(handles[i], inputs_lm[i].device_mem);
    bm_free_device(handles[i], outputs_lm[i].device_mem);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; j++) {
      bm_free_device(handles[j], past_key[i][j].device_mem);
      bm_free_device(handles[j], past_value[i][j].device_mem);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void LLama2Impl::step_back(const bm_tensor_t &kv, const bm_tensor_t &kv_cache) {
  if (token_length >= MAX_LEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / MAX_LEN;
  auto real_size = (token_length - 1) * bytes;
  auto offset = (MAX_LEN - token_length + 1) * bytes;
  auto mem = bm_mem_from_device(bm_mem_get_device_addr(kv.device_mem) + offset,
                                real_size);
  auto mem_cache =
      bm_mem_from_device(bm_mem_get_device_addr(kv_cache.device_mem), bytes);
  auto buffer = new uint8_t[real_size];
  auto buffer_cache = new uint8_t[bytes];
  auto dst = new uint8_t[total_size];
  bm_memcpy_d2s(bm_handle, (void *)buffer, mem);
  bm_memcpy_d2s(bm_handle, (void *)buffer_cache, mem_cache);
  // memset(dst, 0, total_size - real_size);
  memcpy(dst + total_size - real_size - bytes, buffer, real_size);
  memcpy(dst + total_size - bytes, buffer_cache, bytes);
  bm_memcpy_s2d(bm_handle, kv.device_mem, (void *)dst);
  delete[] buffer;
  delete[] buffer_cache;
  delete[] dst;
}

int LLama2Impl::forward_first(const std::vector<int> &tokens) {
  int input_ids[MAX_LEN] = {0}; // start token
  int position_id[MAX_LEN] = {0};
  float attention_mask[MAX_LEN * MAX_LEN] = {0};
  token_length = tokens.size();

  std::copy(tokens.begin(), tokens.end(), input_ids);
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }

  for (int i = 0; i < MAX_LEN; i++) {
    for (int j = 0; j < MAX_LEN; j++) {
      if (j <= i && i < token_length) {
      } else {
        attention_mask[i * MAX_LEN + j] = ATTENTION_MASK;
      }
    }
  }

  // forward embeding
  std::vector<int> input_nums(device_num, 1);
  std::vector<void *> datas(device_num, (void *)input_ids);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed_512.data(), datas.data(),
                           input_nums.data(), device_num);
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), inputs_embed_512.data(),
                            inputs_embed_512.size(), outputs_embed_512.data(),
                            outputs_embed_512.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void *> pos_id_datas(device_num, (void *)position_id);
  std::vector<void *> in_attn_datas(device_num, (void *)attention_mask);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_pid.data(), pos_id_datas.data(),
                           input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_attention.data(),
                           in_attn_datas.data(), input_nums.data(), device_num);

  auto embed_512 = outputs_embed_512;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    embed_512[i].shape = net_blocks[0]->stages[0].input_shapes[0];
    inputs_block.push_back(embed_512[i]);
    inputs_block.push_back(inputs_pid[i]);
    inputs_block.push_back(inputs_attention[i]);
    outputs_block.push_back(embed_512[i]);
    outputs_block.push_back(past_key[0][i]);
    outputs_block.push_back(past_value[0][i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      outputs_block[1 + j * 3] = past_key[i][j];
      outputs_block[2 + j * 3] = past_value[i][j];
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }

  int bytes = embed_512[0].device_mem.size / MAX_LEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm[0].device_mem, 0,
                     embed_512[0].device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

int LLama2Impl::forward_next(int cur_token) {
  float attention_mask[MAX_LEN + 1] = {0};
  for (int i = token_length - 1; i < MAX_LEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // embedding
  std::vector<bm_tensor_t> inputs_embed;
  std::vector<void *> input_datas;
  std::vector<int> input_nums(device_num, 1);
  for (int i = 0; i < device_num; ++i) {
    inputs_embed.push_back(outputs_lm[i]); // token_id
    inputs_embed[i].shape = net_embed_cache->stages[0].input_shapes[0];
    input_datas.push_back((void *)(&cur_token));
  }
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed.data(), input_datas.data(),
                           input_nums.data(), device_num);
  auto ret = bmrt_launch_tensor_ex(
      p_bmrt, name_embed_cache.c_str(), inputs_embed.data(),
      inputs_embed.size(), inputs_lm.data(), inputs_lm.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // blocks
  std::vector<void *> pid_datas(device_num, &position_id);
  std::vector<void *> attn_datas(device_num, attention_mask);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_pid.data(), pid_datas.data(),
                           input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_attention.data(), attn_datas.data(),
                           input_nums.data(), device_num);
  std::vector<bm_tensor_t> embed_1 = inputs_lm;
  for (int i = 0; i < device_num; ++i) {
    embed_1[i].shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  }
  int bytes = bm_mem_get_device_size(past_key[0][0].device_mem) / MAX_LEN;
  int token_offset = (token_length - 1) * bytes;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    inputs_block.push_back(embed_1[i]);
    inputs_block.push_back(next_pid[i]);
    inputs_block.push_back(next_attention[i]);
    inputs_block.push_back(past_key[0][i]);
    inputs_block.push_back(past_value[0][i]);
    outputs_block.push_back(embed_1[i]);
    outputs_block.push_back(present_key_cache[i]);
    outputs_block.push_back(present_value_cache[i]);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      inputs_block[3 + j * 5] = past_key[i][j];
      inputs_block[4 + j * 5] = past_value[i][j];
      bm_set_device_mem(&outputs_block[1 + j * 3].device_mem, bytes,
                        bm_mem_get_device_addr(past_key[i][j].device_mem) +
                            token_offset);
      bm_set_device_mem(&outputs_block[2 + j * 3].device_mem, bytes,
                        bm_mem_get_device_addr(past_value[i][j].device_mem) +
                            token_offset);
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }

  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

void LLama2TPU::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    std::string sys_config = R"(
            [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n)";
    if (input_str == "exit") {
      break;
    }

    input_str = sys_config + input_str + " [/INST] ";
    if (history == "") {
      input_str = sys_config + "\nQuestion:\n" + input_str + "\nAnswer\n:";
    } else {
      input_str = "\nQuestion:\n" + input_str + "\nAnswer:\n";
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}
void LLama2TPU::answer(std::string_view input_str) {
  // history += ("[Round " + std::to_string(round + 1) + "]\n\n问：" + input_str
  // +
  //             "\n\n答：");
  history = input_str;
  int tok_num = 1;
  std::vector<int> tokens;
  tokenizer.Encode(history, &tokens);
  tokens.insert(tokens.begin(), 1);
  if (tokens.empty()) {
    printf("Sorry: your question is too wierd!!\n");
    history = "";
    round = 0;
    return;
  }
  // make sure token not too large
  if (tokens.size() > MAX_LEN - 10) {
    // reset
    if (round == 0) {
      printf("Error: your question is too large!\n");
      return;
    }
    round = 0;
    history = "";
    answer(input_str);
    return;
  }
  int pre_token = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = inner->forward_first(tokens);
  auto t1 = std::chrono::system_clock::now();

  while (token != eos && inner->token_length < MAX_LEN) {
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    tokenizer.Decode(pre_ids, &pre_word);
    tokenizer.Decode(ids, &word);
    std::string diff = word.substr(pre_word.size());
    history += diff;
    std::cout << diff << std::flush;
    if (inner->token_length < MAX_LEN) {
      inner->token_length++;
    }
    tok_num++;
    token = inner->forward_next(token);
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  printf("\n\nfirst token latency: %f s", (use0.count() * 1e-6));
  printf("\nspeed: %f token/s\n", tok_num / (use1.count() * 1e-6));
  if (inner->token_length >= MAX_LEN) {
    round = 0;
    history = history.substr(history.size() / 2);
  } else {
    history += "\n\n";
    round++;
  }
}

static void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

static std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void processArguments(int argc, char *argv[], std::string &llama_model,
                      std::vector<int> &devices) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"dev_id", required_argument, nullptr, 'd'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:d:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      llama_model = optarg;
      break;
    case 'd':
      devices = parseCascadeDevices(optarg);
      break;
    case '?':
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  // set your bmodel path here
  printf("Demo for LLama2-13B in BM1684X\n");
  std::string llama_model;
  std::vector<int> devices{0};
  processArguments(argc, argv, llama_model, devices);
  if (llama_model.empty()) {
    printf("model path required\n");
    return -1;
  }
  printf("Init Environment ...\n");
  printf("==========================\n");
  auto model = from_pretrained(devices, llama_model);

  model.chat();

  return 0;
}

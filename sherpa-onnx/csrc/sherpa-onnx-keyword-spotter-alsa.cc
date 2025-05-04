// sherpa-onnx/csrc/sherpa-onnx-keyword-spotter-alsa.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/parse-options.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Usage:
  ./bin/sherpa-onnx-keyword-spotter-alsa \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=2 \
    --keywords-file=keywords.txt \
    --chunk-size=1000 \
    --buffer-size=1365 \
    --period-size=340 \
    device_name

Please refer to
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
for a list of pre-trained models to download.

The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and device 0 on that card, please use:

  plughw:3,0

as the device_name.
)usage";
  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::KeywordSpotterConfig config;

  config.Register(&po);
  
  int32_t chunk_size = 170;
  int32_t buffer_size = 1365;
  int32_t period_size = 170;
  
  po.Register("chunk-size", &chunk_size, "Number of samples to process in each chunk. "
             "Larger values reduce overruns but increase latency. Range: 170-1365");
  po.Register("buffer-size", &buffer_size, "ALSA buffer size in frames. Larger values "
             "reduce overruns but increase latency. Default: 1365");
  po.Register("period-size", &period_size, "ALSA period size in frames. Should be "
             "set to 2x of chunk-size for best performance. Default: 170");

  po.Read(argc, argv);

  // 限制参数在有效范围内
  chunk_size = std::min(std::max(chunk_size, 170), 1365);
  buffer_size = std::max(buffer_size, chunk_size * 2);
  period_size = std::max(period_size, chunk_size);
  
  fprintf(stderr, "Using chunk size: %d\n", chunk_size);
  fprintf(stderr, "Using buffer size: %d\n", buffer_size);
  fprintf(stderr, "Using period size: %d\n", period_size);

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }
  sherpa_onnx::KeywordSpotter spotter(config);

  int32_t expected_sample_rate = config.feat_config.sampling_rate;

  std::string device_name = po.GetArg(1);
  sherpa_onnx::Alsa alsa(device_name.c_str(), period_size, buffer_size);
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  std::string last_text;

  auto stream = spotter.CreateStream();

  sherpa_onnx::Display display;

  int32_t keyword_index = 0;

  // 使用配置的chunk尺寸
  int32_t chunk = chunk_size;
  
  // 记录上次处理的时间点和平均处理时间
  struct timespec last_process_time;
  clock_gettime(CLOCK_MONOTONIC, &last_process_time);
  
  double avg_process_time = 0.0;
  int32_t process_count = 0;

  while (!stop) {
    struct timespec begin_time;
    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    
    const std::vector<float> &samples = alsa.Read(chunk);

    if (samples.empty()) {
      fprintf(stderr, ">>> 读取数据为空, pass\n");
      continue;
    }
    
    stream->AcceptWaveform(expected_sample_rate, samples.data(), samples.size());

    while (spotter.IsReady(stream.get())) {
      spotter.DecodeStream(stream.get());

      const auto r = spotter.GetResult(stream.get());
      if (!r.keyword.empty()) {
        display.Print(keyword_index, r.AsJsonString());
        fflush(stderr);
        keyword_index++;

        spotter.Reset(stream.get());
      }
    }
    
    // 测量本次处理时间并更新平均处理时间
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    double process_time = (current_time.tv_sec - begin_time.tv_sec) + 
                        (current_time.tv_nsec - begin_time.tv_nsec) / 1e9;
    
    process_count++;
    avg_process_time = (avg_process_time * (process_count - 1) + process_time) / process_count;
    
    // 每10秒检查一次性能并调整chunk大小
    double elapsed = (current_time.tv_sec - last_process_time.tv_sec) + 
                    (current_time.tv_nsec - last_process_time.tv_nsec) / 1e9;
    
    if (elapsed > 10.0) {
      fprintf(stderr, "Average processing time per chunk: %.3f ms, RTF: %.3f\n", 
              avg_process_time * 1000, avg_process_time * expected_sample_rate / chunk);
      
      // 根据平均处理时间自适应调整chunk大小
      double rtf = avg_process_time * expected_sample_rate / chunk;
      
      if (rtf < 0.7 && chunk < buffer_size / 2) {
        // 处理速度很快，可以增加chunk大小以提高效率
        chunk = std::min(chunk + 100, buffer_size / 2);
        fprintf(stderr, "Processing is fast, increasing chunk size to: %d\n", chunk);
      } else if (rtf > 0.9 && chunk > 200) {
        // 处理速度较慢，减小chunk大小以降低overrun风险
        chunk = std::max(chunk - 50, 200);
        fprintf(stderr, "Processing is slow, decreasing chunk size to: %d\n", chunk);
      }
      
      last_process_time = current_time;
      avg_process_time = 0.0;
      process_count = 0;
    }
  }

  return 0;
}

// sherpa-onnx/csrc/sherpa-onnx-keyword-spotter-alsa.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <fstream>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/parse-options.h"

std::atomic<bool> stop(false);

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

void LogKeyword(const std::string &keyword) {
  const std::string dir_path = "/tmp/open-xiaoai";
  const std::string file_path = "/tmp/open-xiaoai/kws.log";
  
  // 检查目录是否存在，不存在则创建
  struct stat st;
  if (::stat(dir_path.c_str(), &st) == -1) {
    ::mkdir(dir_path.c_str(), 0755);
  }
  
  // 获取当前时间戳（毫秒）
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  
  // 创建日志内容
  std::string log_content = std::to_string(millis) + "@" + keyword;
  
  // 读取文件行数
  int line_count = 0;
  std::ifstream read_file(file_path); 
  if (read_file.is_open()) {
    std::string line;
    while (std::getline(read_file, line)) {
      line_count++;
    }
    read_file.close();
  }
  
  // 如果少于 10 行则追加，否则清空重写
  std::ios_base::openmode mode = std::ios::out;
  if (line_count >= 10) {
    // 清空文件重新开始
    mode |= std::ios::trunc;
  } else {
    // 追加模式
    mode |= std::ios::app;
  }
  
  // 写入文件（写入时锁定文件）
  std::ofstream file;
  int fd = open(file_path.c_str(), O_WRONLY | O_CREAT, 0644);
  if (fd != -1) {
    // 获取文件锁
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    
    if (fcntl(fd, F_SETLK, &fl) != -1) {
      file.open(file_path, mode);
      if (file.is_open()) {
        file << log_content << std::endl;
        file.flush(); 
        file.close();
      }
      
      // 释放文件锁
      fl.l_type = F_UNLCK;
      fcntl(fd, F_SETLK, &fl);
    }
    close(fd);
  }
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
    --chunk-size=1024 \
    --buffer-size=1365 \
    --period-size=170 \
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
  
  int32_t buffer_size = 1365;
  int32_t period_size = 170;
  int32_t chunk_size = 1024;
  
  po.Register("buffer-size", &buffer_size, "ALSA buffer size in frames. Default: 1365");
  po.Register("period-size", &period_size, "ALSA period size in frames. Default: 170");
  po.Register("chunk-size", &chunk_size, "Number of samples to process in each chunk. Default: 1024");

  po.Read(argc, argv);

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  // 限制参数在有效范围内
  buffer_size = std::max(buffer_size, 1365);
  period_size = std::max(period_size, 170);
  chunk_size = std::max(chunk_size, 170);
  
  fprintf(stderr, "Using buffer size: %d\n", buffer_size);
  fprintf(stderr, "Using period size: %d\n", period_size);
  fprintf(stderr, "Using chunk size: %d\n", chunk_size);

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
  
  // 双缓冲实现
  std::vector<float> buffer1, buffer2;
  std::vector<float> *writing_buffer = &buffer1;
  std::vector<float> *processing_buffer = &buffer2;
  std::mutex buffer_mutex;
  std::condition_variable buffer_cv;
  bool buffer_ready = false;
  bool started = false;
  
  // 处理线程
  std::thread processing_thread([&]() {
    while (!stop) {
      std::vector<float> local_buffer;
      
      {
        std::unique_lock<std::mutex> lock(buffer_mutex);
        buffer_cv.wait(lock, [&]{ return buffer_ready || stop; });
        
        if (stop) break;
        
        // 交换指针，避免复制
        local_buffer.swap(*processing_buffer);
        buffer_ready = false;
      }
      
      if (!local_buffer.empty()) {
        
        if(!started){
          started = true;
          LogKeyword("__STARTED__");
        }

        fprintf(stderr, "🔥 Processing buffer size: %d\n", local_buffer.size());
        
        stream->AcceptWaveform(expected_sample_rate, local_buffer.data(), local_buffer.size());
        
        while (spotter.IsReady(stream.get())) {
          spotter.DecodeStream(stream.get());
          
          const auto r = spotter.GetResult(stream.get());
          if (!r.keyword.empty()) {
            display.Print(keyword_index, r.AsJsonString()+"\n");

            LogKeyword(r.keyword);

            fflush(stderr);
            keyword_index++;
            
            spotter.Reset(stream.get());
          }
        }
      }
    }
  });

  // 主线程负责采集音频
  while (!stop) {
    const std::vector<float> &samples = alsa.Read(chunk_size);
    
    writing_buffer->insert(writing_buffer->end(), samples.begin(), samples.end());
    
    if (writing_buffer->size() >= chunk_size) {
      std::unique_lock<std::mutex> lock(buffer_mutex);
      if (!buffer_ready) {
        // 交换缓冲区
        std::swap(writing_buffer, processing_buffer);
        buffer_ready = true;
        lock.unlock();
        buffer_cv.notify_one();
      }
    } else {
      struct timespec ts = {0, 1 * 1000000};  // 1毫秒
      nanosleep(&ts, nullptr);
    }
  }
  
  // 等待处理线程结束
  {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    buffer_ready = true;
  }
  buffer_cv.notify_one();
  processing_thread.join();

  return 0;
}

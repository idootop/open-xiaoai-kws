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
  
  int32_t buffer_size = 1365;
  int32_t period_size = 170;
  int32_t chunk_size = 1024;
  
  po.Register("buffer-size", &buffer_size, "ALSA buffer size in frames. Default: 1365");
  po.Register("period-size", &period_size, "ALSA period size in frames. Default: 170");
  po.Register("chunk-size", &chunk_size, "Number of samples to process in each chunk. Default: 1024");

  po.Read(argc, argv);

  // 限制参数在有效范围内
  buffer_size = std::max(buffer_size, 1365);
  period_size = std::max(period_size, 170);
  chunk_size = std::max(chunk_size, 1024);
  
  fprintf(stderr, "Using buffer size: %d\n", buffer_size);
  fprintf(stderr, "Using period size: %d\n", period_size);
  fprintf(stderr, "Using chunk size: %d\n", chunk_size);

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
  
  const std::vector<float> temp_samples;

  while (!stop) {
    const std::vector<float> &samples = alsa.Read(1024);

    temp_samples.insert(temp_samples.end(), samples.begin(), samples.end());

    if (temp_samples.size() < chunk_size) {
      nanosleep(1 * 1000000, 0); // 等待 1ms
      continue;
    }

    stream->AcceptWaveform(expected_sample_rate, temp_samples.data(), temp_samples.size());
    temp_samples.clear();

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
  }

  return 0;
}

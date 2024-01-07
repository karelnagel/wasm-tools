import { useEffect, useRef, useState } from "react";
import { FileInput } from "./segment-anything-react";

const MODELS = {
  tiny_multilingual: {
    base_url: "https://huggingface.co/openai/whisper-tiny/resolve/main/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    title: "Tiny Multilingual",
    size: "151 MB",
  },
  tiny_en: {
    base_url: "https://huggingface.co/openai/whisper-tiny.en/resolve/main/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    title: "Tiny English",
    size: "151 MB",
  },
  tiny_quantized_multilingual_q80: {
    base_url: "https://huggingface.co/lmz/candle-whisper/resolve/main/",
    model: "model-tiny-q80.gguf",
    tokenizer: "tokenizer-tiny.json",
    config: "config-tiny.json",
    title: "Quantized Tiny Multilingual",
    size: "41.5 MB",
  },
  tiny_en_quantized_q80: {
    base_url: "https://huggingface.co/lmz/candle-whisper/resolve/main/",
    model: "model-tiny-q80.gguf",
    tokenizer: "tokenizer-tiny-en.json",
    config: "config-tiny-en.json",
    title: "Quantized Tiny English",
    size: "41.8 MB",
  },
  distil_medium_en: {
    base_url: "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    title: "Distil Medium English",
    size: "789 MB",
  },
};

const useWhisperWorker = () => {
  const whisperWorker = useRef<Worker | null>(null);
  useEffect(() => {
    whisperWorker.current = new Worker("/wasm-tools/whisper/worker.js", {
      type: "module",
    });
    return () => {
      whisperWorker.current?.terminate();
    };
  }, []);
  const classifyAudio = (
    weightsURL: string,
    modelID: keyof typeof MODELS,
    tokenizerURL: string,
    configURL: string,
    mel_filtersURL: string,
    audioURL: string,
    setStatus: (status: string) => void
  ): Promise<{ status: string; message: string; output: any }> => {
    return new Promise((resolve, reject) => {
      whisperWorker.current?.postMessage({
        weightsURL,
        modelID,
        tokenizerURL,
        configURL,
        mel_filtersURL,
        audioURL,
      });
      function messageHandler(event: any) {
        console.log("eventtt", event.data);
        if ("status" in event.data) {
          setStatus(event.data.status);
        }
        if ("error" in event.data) {
          whisperWorker.current?.removeEventListener("message", messageHandler);
          reject(new Error(event.data.error));
        }
        if (event.data.status === "complete") {
          whisperWorker.current?.removeEventListener("message", messageHandler);
          resolve(event.data);
        }
      }
      whisperWorker.current?.addEventListener("message", messageHandler);
    });
  };
  return { whisper: whisperWorker.current, classifyAudio };
};

export const WhisperReact = () => {
  const [modelId, setModelId] = useState<keyof typeof MODELS>("tiny_multilingual");
  const [status, setStatus] = useState<string>();
  const [audioUrl, setAudioUrl] = useState<string>();
  const [output, setOutput] = useState<string>();
  const { classifyAudio } = useWhisperWorker();

  const transcribe = async () => {
    if (!audioUrl) return;
    const model = MODELS[modelId];
    const modelUrl = model.base_url + model.model;
    const tokenizerUrl = model.base_url + model.tokenizer;
    const configUrl = model.base_url + model.config;
    const result = await classifyAudio(
      modelUrl,
      modelId,
      tokenizerUrl,
      configUrl,
      "https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors",
      audioUrl,
      setStatus
    );
    const output = result.output.map((segment: any) => segment.dr.text).join(" ");
    setOutput(output);
  };
  return (
    <div className="w-full flex flex-col gap-2">
      <div className="flex items-center gap-3 justify-center">
        <p className="">Model:</p>
        <select onChange={(e) => setModelId(e.target.value as any)} value={modelId}>
          {Object.entries(MODELS).map(([key, value]) => (
            <option key={key} value={key}>
              {value.title} ({value.size})
            </option>
          ))}
        </select>
      </div>

      {!audioUrl && (
        <div className="relative h-[250px] w-full border-2 border-dotted">
          <FileInput
            type="audio"
            accept="audio/*"
            onChange={(file) => {
              setAudioUrl(URL.createObjectURL(file));
            }}
          />
        </div>
      )}

      {audioUrl && <audio src={audioUrl} controls className="h-6 w-full" />}
      <p>{status}</p>
      <button disabled={!audioUrl || status === "decoding" || status === "loading"} onClick={transcribe} className="button bg-black text-white">
        Transcribe
      </button>
      {output && <textarea readOnly className="border w-full h-32 bg-transparent" value={output} />}
    </div>
  );
};

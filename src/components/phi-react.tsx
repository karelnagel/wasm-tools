import { cx } from "class-variance-authority";
import { useEffect, useRef, useState } from "react";

const MODELS = {
  puffin_phi_v2_q4k: {
    base_url: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/",
    model: "model-puffin-phi-v2-q4k.gguf",
    tokenizer: "tokenizer-puffin-phi-v2.json",
    config: "puffin-phi-v2.json",
    quantized: true,
    seq_len: 2048,
    size: "798 MB",
  },
  puffin_phi_v2_q80: {
    base_url: "https://huggingface.co/lmz/candle-quantized-phi/resolve/main/",
    model: "model-puffin-phi-v2-q80.gguf",
    tokenizer: "tokenizer-puffin-phi-v2.json",
    config: "puffin-phi-v2.json",
    quantized: true,
    seq_len: 2048,
    size: "1.50 GB",
  },
  phi_2_0_q4k: {
    base_url: "https://huggingface.co/radames/phi-2-quantized/resolve/main/",
    model: ["model-v2-q4k.gguf_aa.part", "model-v2-q4k.gguf_ab.part", "model-v2-q4k.gguf_ac.part"],
    tokenizer: "tokenizer.json",
    config: "config.json",
    quantized: true,
    seq_len: 2048,
    size: "1.57GB",
  },
};
type Data = {
  status: string;
  error?: string;
  output?: string;
  token?: string;
  sentence?: string;
  tokensSec?: number;
  totalTime?: number;
};
const usePhiWorker = (modelId: keyof typeof MODELS) => {
  const phiWorker = useRef<Worker>();
  const [status, setStatus] = useState<string>();
  const [output, setOutput] = useState<string>();
  const controller = useRef<AbortController>();
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    phiWorker.current = new Worker("/wasm-tools/phi/worker.js", { type: "module" });
    controller.current = new AbortController();
  }, []);

  const abort = () => {
    controller.current!.abort();
    controller.current = new AbortController();
  };

  async function generateSequence(prompt: string, callback: (sentence: string) => void): Promise<Data> {
    if (isRunning) abort();
    setIsRunning(true);

    const model = MODELS[modelId];
    const weightsURL = model.model instanceof Array ? model.model.map((m) => model.base_url + m) : model.base_url + model.model;
    const tokenizerURL = model.base_url + model.tokenizer;
    const configURL = model.base_url + model.config;

    const temperature = 0;
    const topP = 1;
    const repeatPenalty = 1.1;
    const seed = Math.floor(Math.random() * 1000000);
    const maxSeqLen = 200;

    const res = await new Promise<Data>((resolve, reject) => {
      if (!phiWorker.current) return reject(new Error("Worker not initialized"));
      phiWorker.current.postMessage({
        weightsURL,
        modelId,
        tokenizerURL,
        configURL,
        quantized: model.quantized,
        prompt,
        temp: temperature,
        top_p: topP,
        repeatPenalty,
        seed: seed,
        maxSeqLen,
        command: "start",
      });

      const handleAbort = () => {
        phiWorker.current!.postMessage({ command: "abort" });
      };

      const handleMessage = (event: { data: Data }) => {
        const { status, error } = event.data;
        console.log(event.data);
        if (status) setStatus(status);
        if (error) {
          phiWorker.current!.removeEventListener("message", handleMessage);
          reject(new Error(error));
        }
        if (status === "aborted") {
          phiWorker.current!.removeEventListener("message", handleMessage);
          resolve(event.data);
        }
        if (status === "generating") {
          if (!event.data.sentence) return;
          callback(event.data.sentence.replaceAll("<|endoftext|>", ""));
        }
        if (status === "complete") {
          phiWorker.current!.removeEventListener("message", handleMessage);
          resolve(event.data);
        }
      };

      controller.current!.signal.addEventListener("abort", handleAbort);
      phiWorker.current.addEventListener("message", handleMessage);
    });
    setIsRunning(false);
    setOutput(res.output);
    return res;
  }
  return { generateSequence, status, output };
};

type Message = {
  from: "user" | "assistant";
  text: string;
};

const getPrompt = (messages: Message[]) => {
  const prompt = messages.map((m) => `${m.from.toUpperCase()}: ${m.text}`).join("\n");
  return prompt;
};

export const PhiReact = () => {
  const [modelId, setModelId] = useState<keyof typeof MODELS>("puffin_phi_v2_q4k");
  const { generateSequence, status } = usePhiWorker(modelId);
  const [messages, setMessages] = useState<Message[]>([{ from: "assistant", text: "Hello! How can I assist you today?" }]);
  const [message, setMessage] = useState("");

  const submit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const newMessages = [...messages, { from: "user" as const, text: message }, { from: "assistant" as const, text: "" }];
    setMessages(newMessages);
    setMessage("");
    const prompt = getPrompt(newMessages);
    const index = newMessages.length - 1;
    const generate = (sentence: string) => setMessages((messages) => messages.map((m, i) => (i === index ? { ...m, text: sentence } : m)));
    await generateSequence(prompt, generate);
  };

  return (
    <div className="border rounded-md w-full flex flex-col overflow-hidden">
      <div className="flex justify-between items-center p-2 border-b">
        <div className="flex gap-2 items-center">
          Model:
          <select className="bg-transparent border rounded-md p-1" value={modelId} onChange={(e) => setModelId(e.target.value as any)}>
            {Object.entries(MODELS).map(([modelId, model]) => (
              <option key={modelId} value={modelId}>
                {modelId} ({model.size})
              </option>
            ))}
          </select>
        </div>
        <p>{status}</p>
      </div>

      <div className="flex-col-reverse items-center flex h-[300px] p-2 gap-2">
        {[...messages].reverse().map((m, i) => (
          <div
            key={i}
            className={cx(
              "flex gap-2 rounded-[1em] p-1 px-3 max-w-[80%]",
              m.from === "user" ? "bg-blue-500 text-white ml-auto" : "mr-auto bg-black/10"
            )}
          >
            <p>{m.text}</p>
          </div>
        ))}
      </div>

      <form onSubmit={submit} className="flex gap-2 items-center border-t">
        <input
          type="text"
          className="w-full p-1 px-3 bg-transparent h-9 focus:outline-none"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
        <button type="submit" className="bg-black text-white button">
          Send
        </button>
      </form>
    </div>
  );
};

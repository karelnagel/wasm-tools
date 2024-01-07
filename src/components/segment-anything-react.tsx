import { cx } from "class-variance-authority";
import { Download, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type Point = [number, number, boolean];
const MODEL_BASEURL = "https://huggingface.co/lmz/candle-sam/resolve/main/";
const MODELS = {
  sam_mobile_tiny: {
    url: "mobile_sam-tiny-vitt.safetensors",
    title: "Mobile SAM Tiny (40.6 MB)",
  },
  sam_base: {
    url: "sam_vit_b_01ec64.safetensors",
    title: "SAM Base (375 MB)",
  },
};

export const useSegmentAnything = () => {
  const samWorker = useRef<Worker>();

  useEffect(() => {
    samWorker.current = new Worker("/wasm-tools/segment-anything/worker.js", { type: "module" });
    return () => samWorker.current?.terminate();
  }, []);

  const segmentPoints = async (
    modelURL: string,
    modelID: string,
    imageURL: string,
    points: Point[] | undefined,
    setStatus: (status: string) => void
  ): Promise<{ status: "complete"; output: { maskURL: string } } | { status: "complete-embedding" }> => {
    if (!samWorker.current) throw new Error("No worker");
    samWorker.current.postMessage({ modelURL, modelID, imageURL, points });
    return new Promise((resolve, reject) => {
      function messageHandler(event: any) {
        if (!samWorker.current) throw new Error("No worker");
        console.log(event.data);
        if ("status" in event.data) {
          setStatus(event.data.status);
        }
        if ("error" in event.data) {
          samWorker.current.removeEventListener("message", messageHandler);
          reject(new Error(event.data.error));
        } else if (event.data.status === "complete-embedding") {
          samWorker.current.removeEventListener("message", messageHandler);
          resolve({ status: "complete-embedding" });
        } else if (event.data.status === "complete") {
          samWorker.current.removeEventListener("message", messageHandler);
          resolve({ status: "complete", output: event.data.output });
        }
      }
      samWorker.current!.addEventListener("message", messageHandler);
    });
  };
  return { segmentPoints };
};

const getCtx = (ref: React.RefObject<HTMLCanvasElement>) => {
  if (!ref.current) throw new Error("No canvas");
  const ctx = ref.current.getContext("2d");
  if (!ctx) throw new Error("No ctx");
  return ctx;
};

export const SegmentAnythingReact = () => {
  const [imageUrl, setImageUrl] = useState<string>();
  const [maskUrl, setMaskUrl] = useState<string>();
  const [modelId, setModelId] = useState<keyof typeof MODELS>("sam_mobile_tiny");
  const [status, setStatus] = useState<string>();
  const [pointModeMask, setPointModeMask] = useState<boolean>(true);
  const [pointArr, setPointArr] = useState<Point[]>([]);
  const { segmentPoints } = useSegmentAnything();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement>(null);

  const uploadImage = async (file: File) => {
    const url = URL.createObjectURL(file);
    console.log(url);
    setImageUrl(url);
    clearImageCanvas();
    drawImageCanvas(url);
    setImageEmbeddings(url);
    setPointModeMask(true);
  };

  const clear = () => {
    clearImageCanvas();
    setPointModeMask(true);
    setMaskUrl(undefined);
    setImageUrl(undefined);
    setPointArr([]);
  };

  const download = async () => {
    const loadImageAsync = (imageURL: string): Promise<HTMLImageElement> => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.crossOrigin = "anonymous";
        img.src = imageURL;
      });
    };
    const originalImage = await loadImageAsync(imageUrl!);
    const maskImage = await loadImageAsync(maskUrl!);

    // create main a board to draw
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = originalImage.width;
    canvas.height = originalImage.height;

    // Perform the mask operation
    ctx.drawImage(maskImage, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.drawImage(originalImage, 0, 0);

    // to blob
    const blob: any = await new Promise((resolve) => canvas.toBlob(resolve));
    const resultURL = URL.createObjectURL(blob);

    // download
    const link = document.createElement("a");
    link.href = resultURL;
    link.download = "cutout.png";
    link.click();
  };

  const canvasClick = async (event: any) => {
    if (!canvasRef.current || !maskRef.current) throw new Error("No canvas or mask");
    const targetBox = event.target.getBoundingClientRect();
    const x = (event.clientX - targetBox.left) / targetBox.width;
    const y = (event.clientY - targetBox.top) / targetBox.height;
    const { width, height } = canvasRef.current;
    const ptsToRemove: number[] = [];
    for (const [idx, pts] of pointArr.entries()) {
      const d = Math.sqrt((pts[0] - x) ** 2 + (pts[1] - y) ** 2);
      if (d < 6 / targetBox.width) {
        ptsToRemove.push(idx);
      }
    }
    let newPointsArr: Point[];
    if (ptsToRemove.length > 0) newPointsArr = pointArr.filter((_, idx) => !ptsToRemove.includes(idx));
    else newPointsArr = [...pointArr, [x, y, pointModeMask]];
    if (newPointsArr.length == 0) {
      getCtx(maskRef).clearRect(0, 0, width, height);
      return;
    }
    const { maskURL } = await getSegmentationMask(newPointsArr);
    setMaskUrl(maskURL);
    drawMask(maskURL, newPointsArr);
    setPointArr(newPointsArr);
  };
  const getSegmentationMask = async (points: Point[]) => {
    const modelURL = MODEL_BASEURL + MODELS[modelId].url;
    const res = await segmentPoints(modelURL, modelId, imageUrl!, points, setStatus);
    if (res.status === "complete-embedding") throw new Error("Wrong status");
    return res.output;
  };
  const setImageEmbeddings = async (imageURL: string) => {
    const modelURL = MODEL_BASEURL + MODELS[modelId].url;
    await segmentPoints(modelURL, modelId, imageURL, undefined, setStatus);
  };

  const clearImageCanvas = () => {
    if (!canvasRef.current || !maskRef.current) throw new Error("No canvas or mask");
    const { width, height } = canvasRef.current!;
    getCtx(canvasRef).clearRect(0, 0, width, height);
    getCtx(maskRef).clearRect(0, 0, width, height);
    setPointArr([]);
  };

  function drawMask(maskURL: string, points: Point[]) {
    if (!maskURL) throw new Error("No mask URL provided");
    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      if (!canvasRef.current || !maskRef.current) throw new Error("No canvas or mask");
      maskRef.current.width = canvasRef.current.width;
      maskRef.current.height = canvasRef.current.height;
      const ctxMask = getCtx(maskRef);
      ctxMask.save();
      ctxMask.drawImage(canvasRef.current, 0, 0);
      ctxMask.globalCompositeOperation = "source-atop";
      ctxMask.fillStyle = "rgba(255, 0, 0, 0.6)";
      ctxMask.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      ctxMask.globalCompositeOperation = "destination-in";
      ctxMask.drawImage(img, 0, 0);
      ctxMask.globalCompositeOperation = "source-over";
      for (const pt of points) {
        if (pt[2]) {
          ctxMask.fillStyle = "rgba(0, 255, 255, 1)";
        } else {
          ctxMask.fillStyle = "rgba(255, 255, 0, 1)";
        }
        ctxMask.beginPath();
        ctxMask.arc(pt[0] * canvasRef.current.width, pt[1] * canvasRef.current.height, 3, 0, 2 * Math.PI);
        ctxMask.fill();
      }
      ctxMask.restore();
    };
    img.src = maskURL;
  }

  function drawImageCanvas(imgURL: string) {
    if (!imgURL) throw new Error("No image URL provided");
    if (!canvasRef.current) throw new Error("No canvas");
    getCtx(canvasRef).clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      canvasRef.current!.width = img.width;
      canvasRef.current!.height = img.height;
      getCtx(canvasRef).drawImage(img, 0, 0);
    };
    img.src = imgURL;
  }
  return (
    <div className="flex flex-col gap-2 items-center w-full">
      {!imageUrl && (
        <div className="flex items-center gap-3 ">
          <p className="whitespace-nowrap">Model:</p>
          <select
            value={modelId}
            onChange={(e) => {
              setModelId(e.target.value as keyof typeof MODELS);
            }}
          >
            {Object.entries(MODELS).map(([key, value]) => (
              <option key={key} value={key}>
                {value.title}
              </option>
            ))}
          </select>
        </div>
      )}
      {!!imageUrl && (
        <div className="flex justify-between gap-2 w-full items-center">
          <p>Status: {status}</p>
          <div className="flex items-center gap-4">
            <button onClick={() => setPointModeMask(!pointModeMask)} className="border rounded-md p-1">
              {pointModeMask ? "Mask Point" : "Background Point"}
            </button>
            <button className="text-red-500 rounded-full" onClick={clear}>
              <X />
            </button>
          </div>
        </div>
      )}
      <div className="border-2 border-gray-300 border-dashed relative w-full overflow-hidden">
        <canvas onClick={canvasClick} ref={canvasRef} className=" w-full"></canvas>
        <canvas ref={maskRef} className="pointer-events-none top-0 left-0 absolute w-full"></canvas>
        {!imageUrl && <FileInput onChange={(file) => uploadImage(file)} />}
        {imageUrl && !status?.startsWith("complete") && <LoadingIndicator />}
      </div>
      {status?.startsWith("complete") && (
        <button onClick={download} className="button bg-black text-white">
          <Download />
          Download
        </button>
      )}
    </div>
  );
};

export const FileInput = ({ onChange, type = "image", accept = "image/*" }: { onChange: (e: File) => void; type?: string; accept?: string }) => {
  const [hover, setHover] = useState<boolean>(false);
  const uploadImage = (e: any) => {
    const file = e.target.files[0];
    if (!file) return;
    onChange(file);
  };
  return (
    <label
      htmlFor="file-upload"
      className={cx("flex flex-col items-center justify-center gap-1 h-full absolute w-full top-0 cursor-pointer", hover && "bg-blue-500 text-white")}
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        setHover(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setHover(false);
        const file = e.dataTransfer.files[0];
        if (!file) return;
        onChange(file);
      }}
    >
      <span>Drag and drop your {type} here</span>
      <span>or</span>
      <span>Click to upload</span>
      <input accept={accept} id="file-upload" name="file-upload" onChange={uploadImage} type="file" className="sr-only" />
    </label>
  );
};

const LoadingIndicator = () => {
  return (
    <div className="absolute top-0 left-0 w-full h-full bg-blue-500/20 text-white flex items-center justify-center">
      <Loader2 className="animate-spin h-10 w-10" />
    </div>
  );
};

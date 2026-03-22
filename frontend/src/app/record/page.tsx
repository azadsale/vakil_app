"use client";

import { useState, useRef } from "react";
import {
  Mic, MicOff, Upload, Loader2, CheckCircle, AlertCircle,
  ArrowRight, FileText, Image, FileType,
} from "lucide-react";
import Link from "next/link";

const API_URL = "";

type RecordingState = "idle" | "recording" | "recorded" | "uploading" | "done" | "error";
type InputMode = "voice" | "document";

interface TranscriptResult {
  statement_id: string;
  status: string;
  language_detected: string;
  duration_seconds: number | null;
  transcript_preview: string;
  transcript_length: number;
  extraction_method?: string;
  char_count?: number;
}

export default function RecordPage() {
  const [inputMode, setInputMode] = useState<InputMode>("voice");

  // Voice state
  const [state, setState] = useState<RecordingState>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<TranscriptResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState("mr-IN");
  const [recordingTime, setRecordingTime] = useState(0);

  // Document state
  const [docFile, setDocFile] = useState<File | null>(null);
  const [docState, setDocState] = useState<RecordingState>("idle");
  const [ocrProgress, setOcrProgress] = useState<{ current: number; total: number } | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const languageOptions = [
    { value: "mr-IN", label: "मराठी (Marathi)" },
    { value: "hi-IN", label: "हिन्दी (Hindi)" },
    { value: "en-IN", label: "English (Indian)" },
  ];

  // ── Voice recording ──────────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setState("recorded");
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start(500);
      setState("recording");
      setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime((p) => p + 1), 1000);
    } catch {
      setError("Microphone access denied. Please allow microphone permissions.");
      setState("error");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const handleAudioFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) { setAudioBlob(file); setState("recorded"); }
  };

  const transcribe = async () => {
    if (!audioBlob) return;
    setState("uploading");
    setError(null);

    const formData = new FormData();
    formData.append("audio_file", audioBlob, "recording.webm");
    formData.append("language_code", language);

    try {
      const res = await fetch(`${API_URL}/api/v1/drafting/transcribe`, {
        method: "POST", body: formData,
      });
      if (!res.ok) {
        let msg = "Transcription failed";
        try { const err = await res.json(); msg = err.detail || msg; }
        catch { /* non-JSON error */ }
        throw new Error(msg);
      }
      const data: TranscriptResult = await res.json();
      setResult(data);
      setState("done");
      sessionStorage.setItem("statement_id", data.statement_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setState("error");
    }
  };

  // ── Document upload ───────────────────────────────────────────────────────
  const handleDocFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) { setDocFile(file); setDocState("recorded"); setError(null); }
  };

  const extractDocument = async () => {
    if (!docFile) return;
    setDocState("uploading");
    setOcrProgress(null);
    setError(null);

    const formData = new FormData();
    formData.append("document_file", docFile, docFile.name);
    formData.append("language_code", language);

    try {
      const res = await fetch(`${API_URL}/api/v1/drafting/upload-document`, {
        method: "POST", body: formData,
      });
      if (!res.ok) {
        let msg = "Document extraction failed";
        try { const err = await res.json(); msg = err.detail || msg; }
        catch { const text = await res.text(); if (text.length < 200) msg = text; }
        throw new Error(msg);
      }
      const { job_id } = await res.json();

      // Poll for progress every 2 seconds
      await new Promise<void>((resolve, reject) => {
        pollRef.current = setInterval(async () => {
          try {
            const poll = await fetch(`${API_URL}/api/v1/drafting/ocr-status/${job_id}`);
            const data = await poll.json();

            if (data.status === "processing") {
              if (data.total_pages > 0) {
                setOcrProgress({ current: data.current_page, total: data.total_pages });
              }
            } else if (data.status === "done") {
              clearInterval(pollRef.current!);
              setResult({
                statement_id: data.statement_id,
                status: "transcribed",
                language_detected: data.language_detected,
                duration_seconds: null,
                transcript_preview: data.transcript_preview,
                transcript_length: data.transcript_length,
                extraction_method: data.extraction_method,
                char_count: data.char_count,
              });
              sessionStorage.setItem("statement_id", data.statement_id);
              setDocState("done");
              setState("done");
              resolve();
            } else {
              clearInterval(pollRef.current!);
              reject(new Error(data.error || "OCR failed"));
            }
          } catch (e) {
            clearInterval(pollRef.current!);
            reject(e);
          }
        }, 2000);
      });

    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setDocState("error");
    }
  };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  const docMethodLabel: Record<string, string> = {
    pdf: "PDF text extracted",
    docx: "Word document extracted",
    image: "Image OCR completed",
  };

  const accepted = ".pdf,.docx,.doc,.jpg,.jpeg,.png,.tiff";

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div>
        <p className="text-sm text-slate-400 uppercase tracking-wider font-medium">Step 1 of 3</p>
        <h1 className="text-2xl font-bold text-slate-900 mt-1">Record Client Statement</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Capture the client's statement — by voice recording or by uploading a written document.
        </p>
      </div>

      {/* Input mode toggle */}
      <div className="flex rounded-xl border border-slate-200 overflow-hidden">
        <button
          onClick={() => { setInputMode("voice"); setError(null); }}
          className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-colors ${
            inputMode === "voice"
              ? "bg-blue-600 text-white"
              : "bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          <Mic size={16} /> Voice Recording
        </button>
        <button
          onClick={() => { setInputMode("document"); setError(null); }}
          className={`flex-1 flex items-center justify-center gap-2 py-3 text-sm font-medium transition-colors ${
            inputMode === "document"
              ? "bg-blue-600 text-white"
              : "bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          <FileText size={16} /> Written Statement
        </button>
      </div>

      {/* Language selector */}
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Client's primary language
        </label>
        <div className="flex gap-3">
          {languageOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setLanguage(opt.value)}
              className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${
                language === opt.value
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-slate-700 border-slate-300 hover:border-blue-400"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── VOICE MODE ── */}
      {inputMode === "voice" && state !== "done" && (
        <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center space-y-6">
          {(state === "idle" || state === "error") && (
            <>
              <div className="flex flex-col items-center gap-4">
                <button
                  onClick={startRecording}
                  className="w-24 h-24 rounded-full bg-blue-600 hover:bg-blue-700 text-white flex items-center justify-center shadow-lg hover:shadow-xl transition-all"
                >
                  <Mic size={36} />
                </button>
                <p className="text-slate-500 text-sm">Click to start recording</p>
              </div>
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-slate-200" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="bg-white px-3 text-slate-400">or upload audio file</span>
                </div>
              </div>
              <label className="inline-flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg cursor-pointer text-sm font-medium text-slate-700 transition-colors">
                <Upload size={16} />
                Upload audio file
                <input type="file" accept="audio/*" onChange={handleAudioFileUpload} className="sr-only" />
              </label>
            </>
          )}

          {state === "recording" && (
            <div className="flex flex-col items-center gap-4">
              <div className="relative">
                <button
                  onClick={stopRecording}
                  className="w-24 h-24 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center justify-center shadow-lg animate-pulse"
                >
                  <MicOff size={36} />
                </button>
                <div className="absolute -top-2 -right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full">LIVE</div>
              </div>
              <p className="text-2xl font-mono font-bold text-slate-800">{formatTime(recordingTime)}</p>
              <p className="text-slate-500 text-sm">Recording... click to stop</p>
            </div>
          )}

          {state === "recorded" && (
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                <CheckCircle className="text-green-600" size={32} />
              </div>
              <p className="font-medium text-slate-800">Audio ready</p>
              <div className="flex gap-3">
                <button
                  onClick={() => { setAudioBlob(null); setState("idle"); }}
                  className="px-4 py-2 border border-slate-300 rounded-lg text-sm text-slate-600 hover:bg-slate-50"
                >
                  Re-record
                </button>
                <button
                  onClick={transcribe}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
                >
                  Transcribe with Sarvam AI
                </button>
              </div>
            </div>
          )}

          {state === "uploading" && (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="animate-spin text-blue-600" size={40} />
              <p className="font-medium text-slate-800">Transcribing with Sarvam AI...</p>
              <p className="text-sm text-slate-500">This may take 10–30 seconds</p>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      )}

      {/* ── DOCUMENT MODE ── */}
      {inputMode === "document" && state !== "done" && (
        <div className="bg-white rounded-2xl border border-slate-200 p-8 space-y-6">
          <div className="text-center space-y-2">
            <p className="text-sm font-medium text-slate-700">Upload the client's written statement</p>
            <p className="text-xs text-slate-400">Supported: PDF, Word (.docx), JPG, PNG — handwritten or typed</p>
          </div>

          {/* Format hints */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: FileType, label: "PDF", sub: "Typed / scanned", color: "text-red-500" },
              { icon: FileText, label: "Word (.docx)", sub: "Digital document", color: "text-blue-500" },
              { icon: Image, label: "JPG / PNG", sub: "Handwritten page", color: "text-green-500" },
            ].map(({ icon: Icon, label, sub, color }) => (
              <div key={label} className="flex flex-col items-center gap-1 p-3 rounded-lg bg-slate-50 border border-slate-100">
                <Icon size={24} className={color} />
                <span className="text-xs font-medium text-slate-700">{label}</span>
                <span className="text-xs text-slate-400">{sub}</span>
              </div>
            ))}
          </div>

          {/* Upload area */}
          <label className={`flex flex-col items-center gap-3 p-8 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
            docFile ? "border-green-400 bg-green-50" : "border-slate-300 hover:border-blue-400 hover:bg-blue-50"
          }`}>
            {docFile ? (
              <>
                <CheckCircle className="text-green-600" size={32} />
                <div className="text-center">
                  <p className="font-medium text-slate-800 text-sm">{docFile.name}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{(docFile.size / 1024).toFixed(0)} KB</p>
                </div>
              </>
            ) : (
              <>
                <Upload className="text-slate-400" size={32} />
                <div className="text-center">
                  <p className="text-sm font-medium text-slate-600">Click to select file</p>
                  <p className="text-xs text-slate-400 mt-0.5">PDF, DOCX, JPG, PNG up to 20MB</p>
                </div>
              </>
            )}
            <input type="file" accept={accepted} onChange={handleDocFileSelect} className="sr-only" />
          </label>

          {docFile && docState !== "uploading" && (
            <div className="flex gap-3">
              <button
                onClick={() => { setDocFile(null); setDocState("idle"); }}
                className="px-4 py-2 border border-slate-300 rounded-lg text-sm text-slate-600 hover:bg-slate-50"
              >
                Remove
              </button>
              <button
                onClick={extractDocument}
                className="flex-1 px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
              >
                Extract Text from Document
              </button>
            </div>
          )}

          {docState === "uploading" && (
            <div className="flex flex-col items-center gap-3 py-4">
              <Loader2 className="animate-spin text-blue-600" size={40} />
              {ocrProgress && ocrProgress.total > 0 ? (
                <div className="w-full space-y-2">
                  <div className="flex justify-between text-sm text-slate-600">
                    <span>OCR in progress — reading handwriting…</span>
                    <span className="font-medium">{ocrProgress.current} / {ocrProgress.total} pages</span>
                  </div>
                  <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${Math.round((ocrProgress.current / ocrProgress.total) * 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-slate-400 text-center">
                    {Math.round((ocrProgress.current / ocrProgress.total) * 100)}% complete — please wait
                  </p>
                </div>
              ) : (
                <>
                  <p className="font-medium text-slate-800">Uploading & analysing document…</p>
                  <p className="text-sm text-slate-500">Detecting text layer — OCR starts if needed</p>
                </>
              )}
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
              <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>
      )}

      {/* ── RESULT (shared for both modes) ── */}
      {state === "done" && result && (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-start gap-3">
            <CheckCircle className="text-green-600 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="font-medium text-green-800">
                {result.extraction_method
                  ? docMethodLabel[result.extraction_method] || "Document processed"
                  : "Transcription Complete"}
              </p>
              <p className="text-sm text-green-700 mt-0.5">
                {result.extraction_method
                  ? <>Characters extracted: <strong>{result.transcript_length}</strong></>
                  : <>Language: <strong>{result.language_detected}</strong> · Characters: <strong>{result.transcript_length}</strong></>
                }
              </p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <p className="text-sm font-medium text-slate-500 mb-2">
              {result.extraction_method ? "Extracted Text Preview" : "Transcript Preview"}
            </p>
            <p className="text-slate-800 leading-relaxed text-sm">{result.transcript_preview}</p>
          </div>

          <Link
            href={`/facts?statement_id=${result.statement_id}`}
            className="flex items-center justify-center gap-2 w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-medium transition-colors"
          >
            Continue to Extract Facts
            <ArrowRight size={18} />
          </Link>
        </div>
      )}
    </div>
  );
}

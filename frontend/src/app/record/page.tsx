"use client";

import { useState, useRef } from "react";
import { Mic, MicOff, Upload, Loader2, CheckCircle, AlertCircle, ArrowRight } from "lucide-react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type RecordingState = "idle" | "recording" | "recorded" | "uploading" | "done" | "error";

interface TranscriptResult {
  statement_id: string;
  status: string;
  language_detected: string;
  duration_seconds: number | null;
  transcript_preview: string;
  transcript_length: number;
}

/**
 * Step 1: Record page.
 * Records client's voice via browser MediaRecorder API (WebM/Opus).
 * Uploads to /api/v1/drafting/transcribe → Sarvam Saaras v3.
 * Shows transcript preview and stores statement_id in sessionStorage.
 */
export default function RecordPage() {
  const [state, setState] = useState<RecordingState>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [result, setResult] = useState<TranscriptResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState("mr-IN");
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

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

      recorder.start(500); // collect data every 500ms
      setState("recording");
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setError("Microphone access denied. Please allow microphone permissions.");
      setState("error");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioBlob(file);
      setState("recorded");
    }
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
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Transcription failed");
      }

      const data: TranscriptResult = await res.json();
      setResult(data);
      setState("done");

      // Store statement_id for next steps
      sessionStorage.setItem("statement_id", data.statement_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setState("error");
    }
  };

  const formatTime = (seconds: number) =>
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;

  const languageOptions = [
    { value: "mr-IN", label: "मराठी (Marathi)" },
    { value: "hi-IN", label: "हिन्दी (Hindi)" },
    { value: "en-IN", label: "English (Indian)" },
  ];

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div>
        <p className="text-sm text-slate-400 uppercase tracking-wider font-medium">Step 1 of 3</p>
        <h1 className="text-2xl font-bold text-slate-900 mt-1">Record Client Statement</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Let the client speak. Sarvam AI will transcribe the statement in Marathi, Hindi, or English.
        </p>
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

      {/* Recording controls */}
      {state !== "done" && (
        <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center space-y-6">
          {state === "idle" || state === "error" ? (
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
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="sr-only"
                />
              </label>
            </>
          ) : state === "recording" ? (
            <div className="flex flex-col items-center gap-4">
              <div className="relative">
                <button
                  onClick={stopRecording}
                  className="w-24 h-24 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center justify-center shadow-lg animate-pulse"
                >
                  <MicOff size={36} />
                </button>
                <div className="absolute -top-2 -right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                  LIVE
                </div>
              </div>
              <p className="text-2xl font-mono font-bold text-slate-800">
                {formatTime(recordingTime)}
              </p>
              <p className="text-slate-500 text-sm">Recording... click to stop</p>
            </div>
          ) : state === "recorded" ? (
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
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="animate-spin text-blue-600" size={40} />
              <p className="font-medium text-slate-800">Transcribing with Sarvam AI...</p>
              <p className="text-sm text-slate-500">This may take 10-30 seconds</p>
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

      {/* Result */}
      {state === "done" && result && (
        <div className="space-y-4">
          <div className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-start gap-3">
            <CheckCircle className="text-green-600 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="font-medium text-green-800">Transcription Complete</p>
              <p className="text-sm text-green-700 mt-0.5">
                Language: <strong>{result.language_detected}</strong> · 
                Characters: <strong>{result.transcript_length}</strong>
              </p>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <p className="text-sm font-medium text-slate-500 mb-2">Transcript Preview</p>
            <p className="text-slate-800 leading-relaxed text-sm">
              {result.transcript_preview}
            </p>
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
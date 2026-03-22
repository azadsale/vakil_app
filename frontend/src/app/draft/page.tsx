"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Loader2,
  AlertCircle,
  CheckCircle,
  ThumbsUp,
  ThumbsDown,
  Copy,
  Download,
  AlertTriangle,
} from "lucide-react";

// Empty string = same origin. Next.js rewrites proxy /api/* → http://backend:8000/api/*
const API_URL = "";

interface DraftResult {
  draft_id: string;
  case_id: string;
  status: string;
  version: number;
  draft_preview: string;
  legal_sections_used: string[];
  templates_used_count: number;
  missing_fields: string[];
  created_at: string;
}

interface FullDraft {
  draft_id: string;
  draft_text: string;
  disclaimer: string;
  facts_json: Record<string, unknown>;
  legal_sections_used: string[];
  missing_fields: string[];
  status: string;
}

/**
 * Step 3: Draft review page.
 * Calls /api/v1/drafting/generate → displays full petition.
 * Lawyer can approve, reject, or provide feedback.
 * Copy to clipboard + download as .txt.
 */
function DraftPageContent() {
  const searchParams = useSearchParams();
  const statementId = searchParams.get("statement_id") || sessionStorage.getItem("statement_id") || "";
  const caseId = searchParams.get("case_id") || sessionStorage.getItem("case_id") || "";

  const [language, setLanguage] = useState<"english" | "marathi">("english");
  const [languageSelected, setLanguageSelected] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [draftMeta, setDraftMeta] = useState<DraftResult | null>(null);
  const [fullDraft, setFullDraft] = useState<FullDraft | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackText, setFeedbackText] = useState("");
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackDone, setFeedbackDone] = useState(false);
  const [copied, setCopied] = useState(false);

  const generateDraft = async (lang: "english" | "marathi") => {
    if (!statementId || !caseId) {
      setError("Missing statement ID or case ID");
      return;
    }

    setGenerating(true);
    setError(null);

    const formData = new FormData();
    formData.append("statement_id", statementId);
    formData.append("case_id", caseId);
    formData.append("language", lang);

    // Include pre-extracted facts if available
    const storedFacts = sessionStorage.getItem("facts_json");
    if (storedFacts) {
      formData.append("pre_extracted_facts", storedFacts);
    }

    try {
      const res = await fetch(`${API_URL}/api/v1/drafting/generate`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Draft generation failed");
      }

      const data: DraftResult = await res.json();
      setDraftMeta(data);

      // Fetch full draft text
      const fullRes = await fetch(`${API_URL}/api/v1/drafting/${data.draft_id}`);
      if (fullRes.ok) {
        const fullData: FullDraft = await fullRes.json();
        setFullDraft(fullData);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setGenerating(false);
    }
  };

  const handleLanguageSelect = (lang: "english" | "marathi") => {
    setLanguage(lang);
    setLanguageSelected(true);
    generateDraft(lang);
  };

  useEffect(() => {
    // Do not auto-generate — wait for language selection
  }, []);

  const copyToClipboard = async () => {
    if (!fullDraft?.draft_text) return;
    await navigator.clipboard.writeText(fullDraft.draft_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadAsTxt = () => {
    if (!fullDraft?.draft_text) return;
    const blob = new Blob([fullDraft.draft_text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `DV_Petition_Draft_${language}_${draftMeta?.draft_id?.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const submitFeedback = async (newStatus: "approved" | "rejected") => {
    if (!draftMeta) return;
    setFeedbackSubmitting(true);

    const formData = new FormData();
    formData.append("new_status", newStatus);
    if (feedbackText) formData.append("feedback_notes", feedbackText);

    try {
      const res = await fetch(
        `${API_URL}/api/v1/drafting/${draftMeta.draft_id}/feedback`,
        { method: "PATCH", body: formData }
      );
      if (!res.ok) throw new Error("Feedback submission failed");
      setFeedbackDone(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Feedback failed");
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <p className="text-sm text-slate-400 uppercase tracking-wider font-medium">Step 3 of 3</p>
        <h1 className="text-2xl font-bold text-slate-900 mt-1">Review Generated Draft</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Review the AI-generated petition. Approve to use it as a future style reference, or reject with feedback.
        </p>
      </div>

      {(!statementId || !caseId) && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
          <p>Missing statement or case ID. Please complete Steps 1 and 2 first.</p>
        </div>
      )}

      {/* Language selection — shown before generation starts */}
      {statementId && caseId && !languageSelected && !generating && !draftMeta && (
        <div className="bg-white rounded-xl border border-slate-200 p-8 text-center space-y-6">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">Select Draft Language</h2>
            <p className="text-sm text-slate-500 mt-1">Choose the language for the petition draft</p>
          </div>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => handleLanguageSelect("english")}
              className="flex-1 max-w-xs border-2 border-slate-200 hover:border-blue-500 hover:bg-blue-50 rounded-xl p-6 transition-all group"
            >
              <div className="text-3xl mb-2">🇬🇧</div>
              <div className="font-semibold text-slate-800 group-hover:text-blue-700">English</div>
              <div className="text-xs text-slate-400 mt-1">Standard legal English</div>
            </button>
            <button
              onClick={() => handleLanguageSelect("marathi")}
              className="flex-1 max-w-xs border-2 border-slate-200 hover:border-orange-500 hover:bg-orange-50 rounded-xl p-6 transition-all group"
            >
              <div className="text-3xl mb-2">🇮🇳</div>
              <div className="font-semibold text-slate-800 group-hover:text-orange-700">मराठी</div>
              <div className="text-xs text-slate-400 mt-1">Marathi (Maharashtra courts)</div>
            </button>
          </div>
        </div>
      )}

      {generating && (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-4">
          <Loader2 className="animate-spin text-blue-600 mx-auto" size={48} />
          <div>
            <p className="font-semibold text-slate-800 text-lg">Generating petition draft...</p>
            <p className="text-slate-500 text-sm mt-2">
              Combining: Client facts + DV Act 2005 (RAG) + Lawyer style templates
              {" · "}<span className="font-medium">{language === "marathi" ? "मराठी" : "English"}</span>
            </p>
          </div>
          <div className="flex justify-center gap-6 text-xs text-slate-400">
            <span>⚖️ Querying DV Act sections</span>
            <span>✍️ Matching lawyer style</span>
            <span>📝 Drafting petition</span>
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {draftMeta && !generating && (
        <>
          {/* Metadata bar */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-wrap gap-4 text-sm">
            <div>
              <span className="text-slate-400">Draft ID:</span>{" "}
              <code className="text-slate-700 text-xs bg-white px-1.5 py-0.5 rounded border">
                {draftMeta.draft_id.slice(0, 8)}...
              </code>
            </div>
            <div>
              <span className="text-slate-400">Legal sections:</span>{" "}
              <span className="text-slate-700">{draftMeta.legal_sections_used.length} cited</span>
            </div>
            <div>
              <span className="text-slate-400">Style templates:</span>{" "}
              <span className="text-slate-700">{draftMeta.templates_used_count} used</span>
            </div>
            <div>
              <span className="text-slate-400">Version:</span>{" "}
              <span className="text-slate-700">v{draftMeta.version}</span>
            </div>
            <div>
              <span className="text-slate-400">Language:</span>{" "}
              <span className="text-slate-700">{language === "marathi" ? "मराठी" : "English"}</span>
            </div>
          </div>

          {/* Missing fields warning */}
          {draftMeta.missing_fields.length > 0 && (
            <div className="flex items-start gap-3 bg-orange-50 border border-orange-200 rounded-lg p-4">
              <AlertTriangle className="text-orange-600 flex-shrink-0 mt-0.5" size={18} />
              <div className="text-sm">
                <p className="font-medium text-orange-800">
                  {draftMeta.missing_fields.length} placeholder{draftMeta.missing_fields.length !== 1 ? "s" : ""} in draft
                </p>
                <p className="text-orange-700 mt-1">
                  Fields marked <code className="bg-orange-100 px-1 rounded">[MISSING: ...]</code> must be 
                  filled before filing.
                </p>
              </div>
            </div>
          )}

          {/* Legal sections used */}
          {draftMeta.legal_sections_used.length > 0 && (
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <p className="text-sm font-medium text-blue-800 mb-2">
                ⚖️ Legal grounding — DV Act sections cited
              </p>
              <div className="flex flex-wrap gap-2">
                {draftMeta.legal_sections_used.map((s) => (
                  <span
                    key={s}
                    className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Draft text */}
          {fullDraft && (
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <div className="flex items-center justify-between px-5 py-3 border-b border-slate-200 bg-slate-50">
                <h2 className="font-semibold text-slate-700">Generated Petition</h2>
                <div className="flex gap-2">
                  <button
                    onClick={copyToClipboard}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors"
                  >
                    {copied ? <CheckCircle size={14} className="text-green-600" /> : <Copy size={14} />}
                    {copied ? "Copied!" : "Copy"}
                  </button>
                  <button
                    onClick={downloadAsTxt}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors"
                  >
                    <Download size={14} />
                    Download
                  </button>
                </div>
              </div>
              <div className="p-6 legal-text text-sm text-slate-800 whitespace-pre-wrap max-h-[600px] overflow-y-auto leading-relaxed">
                {fullDraft.draft_text}
              </div>
            </div>
          )}

          {/* Feedback */}
          {!feedbackDone ? (
            <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
              <h3 className="font-semibold text-slate-800">Advocate Review</h3>
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="Optional: Add notes for improvement (e.g., 'Add more detail on economic abuse incidents')"
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 h-24 resize-none"
              />
              <div className="flex gap-3">
                <button
                  onClick={() => submitFeedback("approved")}
                  disabled={feedbackSubmitting}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
                >
                  {feedbackSubmitting ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <ThumbsUp size={16} />
                  )}
                  Approve Draft
                </button>
                <button
                  onClick={() => submitFeedback("rejected")}
                  disabled={feedbackSubmitting}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 rounded-lg font-medium text-sm transition-colors disabled:opacity-50"
                >
                  <ThumbsDown size={16} />
                  Reject & Regenerate
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3 bg-green-50 border border-green-200 rounded-xl p-4">
              <CheckCircle className="text-green-600" size={24} />
              <div>
                <p className="font-medium text-green-800">Feedback submitted</p>
                <p className="text-sm text-green-700">
                  Approved drafts can be promoted to templates to improve future drafts.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function DraftPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center p-12"><Loader2 className="animate-spin" size={32} /></div>}>
      <DraftPageContent />
    </Suspense>
  );
}
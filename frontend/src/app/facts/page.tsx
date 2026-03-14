"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Loader2, AlertCircle, ArrowRight, CheckCircle, AlertTriangle } from "lucide-react";
import Link from "next/link";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface FactsResult {
  statement_id: string;
  facts: Record<string, unknown>;
  missing_fields: string[];
  missing_count: number;
  ready_to_draft: boolean;
}

/**
 * Step 2: Facts review page.
 * Calls /api/v1/drafting/extract-facts for the statement_id.
 * Shows extracted facts and missing fields (shown as orange warnings).
 * User confirms before proceeding to draft generation.
 */
function FactsPageContent() {
  const searchParams = useSearchParams();
  const statementId = searchParams.get("statement_id") || sessionStorage.getItem("statement_id") || "";

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FactsResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [caseId, setCaseId] = useState("");

  const extractFacts = async () => {
    if (!statementId) {
      setError("No statement ID found. Please go back to Step 1 and record a statement first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("statement_id", statementId);

    try {
      const res = await fetch(`${API_URL}/api/v1/drafting/extract-facts`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Fact extraction failed");
      }

      const data: FactsResult = await res.json();
      setResult(data);
      sessionStorage.setItem("facts_json", JSON.stringify(data.facts));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (statementId && !result) {
      extractFacts();
    }
  }, [statementId]);

  const renderValue = (value: unknown, path: string = ""): React.ReactNode => {
    if (typeof value === "string") {
      const isMissing = value.startsWith("[MISSING:");
      return (
        <span className={isMissing ? "text-orange-600 font-medium bg-orange-50 px-1 rounded" : "text-slate-800"}>
          {value}
        </span>
      );
    }
    if (Array.isArray(value)) {
      if (value.length === 0) return <span className="text-slate-400 italic text-sm">None listed</span>;
      return (
        <ul className="space-y-1 mt-1">
          {value.map((item, i) => (
            <li key={i} className="text-sm">
              {typeof item === "object" ? (
                <div className="bg-slate-50 rounded p-2 mt-1">
                  {Object.entries(item as Record<string, unknown>).map(([k, v]) => (
                    <div key={k} className="flex gap-2 text-sm">
                      <span className="text-slate-400 min-w-[120px] font-medium capitalize">
                        {k.replace(/_/g, " ")}:
                      </span>
                      <span>{renderValue(v, `${path}[${i}].${k}`)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                renderValue(item, `${path}[${i}]`)
              )}
            </li>
          ))}
        </ul>
      );
    }
    if (typeof value === "object" && value !== null) {
      return (
        <div className="space-y-1 mt-1">
          {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <div key={k} className="flex gap-2 text-sm">
              <span className="text-slate-400 min-w-[120px] font-medium capitalize">
                {k.replace(/_/g, " ")}:
              </span>
              <span>{renderValue(v, `${path}.${k}`)}</span>
            </div>
          ))}
        </div>
      );
    }
    if (typeof value === "boolean") {
      return <span className={value ? "text-green-600" : "text-red-500"}>{value ? "Yes" : "No"}</span>;
    }
    return <span className="text-slate-800">{String(value)}</span>;
  };

  const SECTION_LABELS: Record<string, string> = {
    petitioner: "👤 Petitioner",
    respondent: "👤 Respondent",
    shared_household: "🏠 Shared Household",
    relationship_details: "💍 Relationship",
    incidents: "⚠️ Incidents of Violence",
    reliefs_sought: "⚖️ Reliefs Sought",
    maintenance_details: "💰 Maintenance",
    previous_legal_proceedings: "📋 Previous Cases",
    additional_facts: "📝 Additional Facts",
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div>
        <p className="text-sm text-slate-400 uppercase tracking-wider font-medium">Step 2 of 3</p>
        <h1 className="text-2xl font-bold text-slate-900 mt-1">Review Extracted Facts</h1>
        <p className="text-slate-500 mt-1 text-sm">
          AI has extracted the following facts from the client's statement. 
          Fields marked in <span className="text-orange-600 font-medium">orange</span> are missing and need to be verified.
        </p>
      </div>

      {!statementId && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">No statement found</p>
            <p className="text-sm mt-1">
              <Link href="/record" className="underline">Go back to Step 1</Link> to record a client statement first.
            </p>
          </div>
        </div>
      )}

      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-3">
          <Loader2 className="animate-spin text-blue-600 mx-auto" size={40} />
          <p className="font-medium text-slate-800">Extracting facts with AI...</p>
          <p className="text-sm text-slate-500">Analyzing client statement for legal facts</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {result && !loading && (
        <>
          {/* Missing fields warning */}
          {result.missing_count > 0 && (
            <div className="flex items-start gap-3 bg-orange-50 border border-orange-200 rounded-lg p-4">
              <AlertTriangle className="text-orange-600 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="font-medium text-orange-800">
                  {result.missing_count} missing field{result.missing_count !== 1 ? "s" : ""} detected
                </p>
                <p className="text-sm text-orange-700 mt-1">
                  These fields were not mentioned by the client. Please verify with the client before generating the draft.
                </p>
                <ul className="mt-2 space-y-0.5">
                  {result.missing_fields.map((f) => (
                    <li key={f} className="text-sm text-orange-700 font-mono">• {f}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {result.ready_to_draft && (
            <div className="flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg p-3 text-green-700">
              <CheckCircle size={18} />
              <span className="text-sm font-medium">All facts complete — ready to generate draft</span>
            </div>
          )}

          {/* Facts display */}
          <div className="space-y-4">
            {Object.entries(result.facts)
              .filter(([key]) => !key.startsWith("_"))
              .map(([section, value]) => (
                <div key={section} className="bg-white rounded-xl border border-slate-200 p-5">
                  <h3 className="font-semibold text-slate-700 mb-3">
                    {SECTION_LABELS[section] || section.replace(/_/g, " ")}
                  </h3>
                  {renderValue(value, section)}
                </div>
              ))}
          </div>

          {/* Case ID input */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Case ID (required for draft generation)
            </label>
            <input
              type="text"
              value={caseId}
              onChange={(e) => setCaseId(e.target.value)}
              placeholder="Enter case UUID or create a new case"
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-slate-400 mt-1">
              From the Cases module. For now, enter any valid UUID for testing.
            </p>
          </div>

          {/* Next step */}
          <Link
            href={`/draft?statement_id=${result.statement_id}&case_id=${caseId}`}
            className={`flex items-center justify-center gap-2 w-full py-3 rounded-xl font-medium transition-colors ${
              caseId
                ? "bg-blue-600 hover:bg-blue-700 text-white"
                : "bg-slate-200 text-slate-400 cursor-not-allowed pointer-events-none"
            }`}
          >
            Generate DV Petition Draft
            <ArrowRight size={18} />
          </Link>
        </>
      )}
    </div>
  );
}

export default function FactsPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center p-12"><Loader2 className="animate-spin" size={32} /></div>}>
      <FactsPageContent />
    </Suspense>
  );
}
"use client";

import React, { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Loader2,
  AlertCircle,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import Link from "next/link";

const API_URL = "";

interface FactsResult {
  statement_id: string;
  facts: Record<string, unknown>;
  missing_fields: string[];
  missing_count: number;
  ready_to_draft: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate a RFC-4122 v4 UUID in the browser. */
function generateUUID(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/** Recursively collect all "leaf" paths in the facts object. */
function collectPaths(
  obj: unknown,
  prefix = ""
): Array<{ path: string; value: unknown }> {
  if (typeof obj !== "object" || obj === null) {
    return [{ path: prefix, value: obj }];
  }
  if (Array.isArray(obj)) {
    return obj.flatMap((item, i) =>
      collectPaths(item, `${prefix}[${i}]`)
    );
  }
  return Object.entries(obj as Record<string, unknown>).flatMap(([k, v]) =>
    collectPaths(v, prefix ? `${prefix}.${k}` : k)
  );
}

/** Set a nested value at a dot-path (e.g. "petitioner.name"). */
function setAtPath(
  obj: Record<string, unknown>,
  path: string,
  value: string
): Record<string, unknown> {
  const clone = JSON.parse(JSON.stringify(obj)) as Record<string, unknown>;
  const parts = path.replace(/\[(\d+)\]/g, ".$1").split(".");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let cur: any = clone;
  for (let i = 0; i < parts.length - 1; i++) {
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
  return clone;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface EditableFieldProps {
  value: unknown;
  path: string;
  onChange: (path: string, newVal: string) => void;
}

/** Single editable leaf value. String fields become <input>. */
function EditableField({ value, path, onChange }: EditableFieldProps) {
  if (typeof value === "string") {
    const isMissing = value.startsWith("[MISSING:");
    return (
      <input
        type="text"
        defaultValue={isMissing ? "" : value}
        placeholder={isMissing ? value : undefined}
        onChange={(e) => onChange(path, e.target.value)}
        className={`w-full border rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
          isMissing
            ? "border-orange-300 bg-orange-50 placeholder-orange-400 text-slate-800"
            : "border-slate-200 bg-white text-slate-800"
        }`}
      />
    );
  }
  if (typeof value === "boolean") {
    return (
      <span className={value ? "text-green-600 text-sm" : "text-red-500 text-sm"}>
        {value ? "Yes" : "No"}
      </span>
    );
  }
  return <span className="text-slate-800 text-sm">{String(value ?? "")}</span>;
}

interface FactsSectionProps {
  sectionKey: string;
  value: unknown;
  editedFacts: Record<string, unknown>;
  onFieldChange: (path: string, newVal: string) => void;
}

/** Render a full section (handles nested objects and arrays recursively). */
function FactsSection({ sectionKey, value, editedFacts, onFieldChange }: FactsSectionProps) {
  // Get the current (possibly edited) version of this section
  const currentValue =
    (editedFacts[sectionKey] as unknown) ?? value;

  function renderNode(node: unknown, prefix: string): React.ReactNode {  // eslint-disable-line
    if (Array.isArray(node)) {
      if (node.length === 0)
        return <span className="text-slate-400 italic text-sm">None listed</span>;
      return (
        <ul className="space-y-2 mt-1">
          {node.map((item, i) => (
            <li key={i} className="bg-slate-50 rounded p-2 border border-slate-100">
              {renderNode(item, `${prefix}[${i}]`)}
            </li>
          ))}
        </ul>
      );
    }
    if (typeof node === "object" && node !== null) {
      return (
        <div className="space-y-2 mt-1">
          {Object.entries(node as Record<string, unknown>).map(([k, v]) => (
            <div key={k} className="grid grid-cols-[140px_1fr] gap-2 items-start">
              <span className="text-slate-500 text-sm font-medium pt-1.5 capitalize">
                {k.replace(/_/g, " ")}
              </span>
              {renderNode(v, `${prefix}.${k}`)}
            </div>
          ))}
        </div>
      );
    }
    // Leaf
    return (
      <EditableField
        value={node}
        path={prefix}
        onChange={onFieldChange}
      />
    );
  }

  return <>{renderNode(currentValue, sectionKey)}</>;
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

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

// Maps JSON field paths → plain-English label the lawyer sees
// Covers both nested paths (petitioner.name) and flat paths (name) for robustness
const FIELD_LABELS: Record<string, string> = {
  // Nested paths (correct schema)
  "petitioner.name":                         "Petitioner's full name",
  "petitioner.age":                          "Petitioner's age",
  "petitioner.address":                      "Petitioner's current address",
  "petitioner.occupation":                   "Petitioner's occupation",
  "respondent.name":                         "Respondent's (husband's) full name",
  "respondent.age":                          "Respondent's age",
  "respondent.address":                      "Respondent's current address",
  "respondent.occupation":                   "Respondent's occupation",
  "respondent.relationship_to_petitioner":   "Relationship to petitioner",
  "shared_household.address":                "Address of shared household",
  "shared_household.ownership":              "Who owns the shared house",
  "shared_household.duration_of_residence":  "How long they lived together",
  "relationship_details.date_of_marriage":   "Date of marriage",
  "relationship_details.place_of_marriage":  "Place of marriage",
  "relationship_details.marriage_type":      "Type of marriage (registered / religious)",
  // Flat paths (if LLM flattens the structure)
  "name":                  "Client's full name",
  "age":                   "Client's age",
  "address":               "Client's address",
  "occupation":            "Client's occupation",
  "date_of_marriage":      "Date of marriage",
  "place_of_marriage":     "Place of marriage",
  "date_of_incident":      "Date of incident",
  "date":                  "Date of incident",
  "incident_description":  "What happened (incident description)",
  "description":           "What happened (incident description)",
  "police_complaint":      "Police complaint / FIR filed?",
  "fir_number":            "FIR number",
  "reliefs_sought":        "Reliefs / orders requested from court",
  "relief":                "Reliefs / orders requested from court",
};

/** Convert a raw JSON path like "petitioner.name" to a friendly label. */
function friendlyLabel(path: string): string {
  // Direct match
  if (FIELD_LABELS[path]) return FIELD_LABELS[path];
  // Incident fields: incidents[0].incident_date → "Incident 1 — date"
  const incidentMatch = path.match(/^incidents\[(\d+)\]\.(.+)$/);
  if (incidentMatch) {
    const idx = parseInt(incidentMatch[1]) + 1;
    const sub = incidentMatch[2].replace(/_/g, " ");
    return `Incident ${idx} — ${sub}`;
  }
  // Children fields
  const childMatch = path.match(/^relationship_details\.children\[(\d+)\]\.(.+)$/);
  if (childMatch) {
    const idx = parseInt(childMatch[1]) + 1;
    const sub = childMatch[2].replace(/_/g, " ");
    return `Child ${idx} — ${sub}`;
  }
  // Fallback: humanise the path
  return path.replace(/_/g, " ").replace(/\./g, " › ").replace(/\[(\d+)\]/g, " $1");
}

function FactsPageContent() {
  const searchParams = useSearchParams();
  const statementId =
    searchParams.get("statement_id") ||
    (typeof sessionStorage !== "undefined"
      ? sessionStorage.getItem("statement_id") ?? ""
      : "");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FactsResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Editable copy of the facts — starts as a clone of the extracted facts
  const [editedFacts, setEditedFacts] = useState<Record<string, unknown>>({});
  // Auto-generate a Case ID so the lawyer can proceed immediately
  const [caseId, setCaseId] = useState<string>(() => generateUUID());

  const extractFacts = useCallback(async () => {
    if (!statementId) {
      setError(
        "No statement ID found. Please go back to Step 1 and record a statement first."
      );
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
      // Deep-clone into editedFacts so user edits are tracked separately
      setEditedFacts(JSON.parse(JSON.stringify(data.facts)));
      sessionStorage.setItem("facts_json", JSON.stringify(data.facts));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [statementId]);

  useEffect(() => {
    if (statementId && !result) {
      extractFacts();
    }
  }, [statementId, result, extractFacts]);

  /** Called by EditableField whenever the lawyer types in a field. */
  const handleFieldChange = useCallback(
    (path: string, newVal: string) => {
      setEditedFacts((prev) => {
        const updated = setAtPath(prev, path, newVal);
        // Persist inside the functional update so we always have the latest state
        // This avoids stale closure issues where sessionStorage misses edits
        sessionStorage.setItem("facts_json", JSON.stringify(updated));
        return updated;
      });
    },
    [] // No dependencies needed — uses functional state update
  );

  // Count remaining [MISSING:] fields in editedFacts
  const remainingMissing = result
    ? collectPaths(editedFacts).filter(
        ({ value }) =>
          typeof value === "string" && value.startsWith("[MISSING:")
      ).length
    : 0;

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <p className="text-sm text-slate-400 uppercase tracking-wider font-medium">
          Step 2 of 3
        </p>
        <h1 className="text-2xl font-bold text-slate-900 mt-1">
          Review &amp; Edit Extracted Facts
        </h1>
        <p className="text-slate-500 mt-1 text-sm">
          AI has extracted the following facts. Fields with{" "}
          <span className="text-orange-600 font-medium">orange</span> borders
          were not mentioned — fill them in before generating the draft.
        </p>
      </div>

      {/* No statement error */}
      {!statementId && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">No statement found</p>
            <p className="text-sm mt-1">
              <Link href="/record" className="underline">
                Go back to Step 1
              </Link>{" "}
              to record a client statement first.
            </p>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-3">
          <Loader2 className="animate-spin text-blue-600 mx-auto" size={40} />
          <p className="font-medium text-slate-800">Extracting facts with AI…</p>
          <p className="text-sm text-slate-500">
            Analyzing client statement for legal facts
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <span>{error}</span>
            <button
              onClick={extractFacts}
              className="mt-2 flex items-center gap-1 text-sm text-red-600 hover:text-red-800 underline"
            >
              <RefreshCw size={13} /> Try again
            </button>
          </div>
        </div>
      )}

      {result && !loading && (
        <>
          {/* Missing fields banner — specific checklist */}
          {remainingMissing > 0 && (
            <div className="bg-orange-50 border border-orange-200 rounded-xl p-4 space-y-3">
              <div className="flex items-start gap-3">
                <AlertTriangle className="text-orange-600 flex-shrink-0 mt-0.5" size={20} />
                <div>
                  <p className="font-semibold text-orange-800">
                    {remainingMissing} field{remainingMissing !== 1 ? "s" : ""} still missing
                    — the draft will show <code className="bg-orange-100 px-1 rounded text-xs">[MISSING]</code> for these
                  </p>
                  <p className="text-xs text-orange-700 mt-1">
                    Fill them in the sections below, or go back to Step 1 and re-record with more detail.
                  </p>
                </div>
              </div>
              {/* Specific list of missing fields */}
              <div className="ml-8 grid gap-1 sm:grid-cols-2">
                {collectPaths(editedFacts)
                  .filter(({ value }) => typeof value === "string" && value.startsWith("[MISSING:"))
                  .map(({ path }) => (
                    <div key={path} className="flex items-center gap-2 text-xs text-orange-800">
                      <span className="w-1.5 h-1.5 rounded-full bg-orange-400 flex-shrink-0" />
                      {friendlyLabel(path)}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {remainingMissing === 0 && (
            <div className="flex items-center gap-2 bg-green-50 border border-green-200 rounded-lg p-3 text-green-700">
              <CheckCircle size={18} />
              <span className="text-sm font-medium">
                All facts complete — ready to generate draft
              </span>
            </div>
          )}

          {/* Editable facts sections */}
          <div className="space-y-4">
            {Object.entries(result.facts)
              .filter(([key]) => !key.startsWith("_"))
              .map(([section, value]) => (
                <div
                  key={section}
                  className="bg-white rounded-xl border border-slate-200 p-5"
                >
                  <h3 className="font-semibold text-slate-700 mb-3">
                    {SECTION_LABELS[section] ||
                      section.replace(/_/g, " ")}
                  </h3>
                  <FactsSection
                    sectionKey={section}
                    value={value}
                    editedFacts={editedFacts}
                    onFieldChange={handleFieldChange}
                  />
                </div>
              ))}
          </div>

          {/* Case ID */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Case ID
            </label>
            <p className="text-xs text-slate-400 mb-2">
              Auto-generated for this session. Replace with an existing Case UUID
              if you have one.
            </p>
            <input
              type="text"
              value={caseId}
              onChange={(e) => setCaseId(e.target.value)}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={() => setCaseId(generateUUID())}
              className="mt-2 text-xs text-blue-600 hover:text-blue-800 underline"
            >
              Generate new UUID
            </button>
          </div>

          {/* Next step — always enabled when caseId is set */}
          <Link
            href={`/draft?statement_id=${result.statement_id}&case_id=${caseId}`}
            onClick={() => {
              sessionStorage.setItem("facts_json", JSON.stringify(editedFacts));
              sessionStorage.setItem("case_id", caseId);
            }}
            className="flex items-center justify-center gap-2 w-full py-3 rounded-xl font-medium transition-colors bg-blue-600 hover:bg-blue-700 text-white"
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
    <Suspense
      fallback={
        <div className="flex items-center justify-center p-12">
          <Loader2 className="animate-spin" size={32} />
        </div>
      }
    >
      <FactsPageContent />
    </Suspense>
  );
}
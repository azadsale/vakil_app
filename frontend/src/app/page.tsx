"use client";

import Link from "next/link";
import { Mic, FileText, CheckCircle, ArrowRight } from "lucide-react";

/**
 * Home page — workflow overview and entry point.
 * Guides the lawyer through the 3-step DV petition drafting process.
 */
export default function HomePage() {
  const steps = [
    {
      number: 1,
      icon: Mic,
      title: "Record Client Statement",
      description:
        "Record the client's verbal statement in Marathi, Hindi, or English. Sarvam AI transcribes it instantly.",
      href: "/record",
      color: "bg-blue-600",
      lightColor: "bg-blue-50 border-blue-200",
    },
    {
      number: 2,
      icon: FileText,
      title: "Review & Complete Facts",
      description:
        "Review extracted facts. Fill in any missing details (shown as placeholders). Verify before generating.",
      href: "/facts",
      color: "bg-amber-600",
      lightColor: "bg-amber-50 border-amber-200",
    },
    {
      number: 3,
      icon: CheckCircle,
      title: "Review & Approve Draft",
      description:
        "AI generates the complete DV petition. Review, edit, and approve. Approved drafts improve future results.",
      href: "/draft",
      color: "bg-green-600",
      lightColor: "bg-green-50 border-green-200",
    },
  ];

  return (
    <div className="space-y-10">
      {/* Hero */}
      <div className="text-center space-y-3 pt-4">
        <h1 className="text-3xl font-bold text-slate-900">
          DV Petition Drafting Assistant
        </h1>
        <p className="text-slate-500 max-w-xl mx-auto">
          Three steps. Client's voice → Structured facts → Court-ready petition.
          Grounded in the Protection of Women from Domestic Violence Act, 2005.
        </p>
      </div>

      {/* Step Cards */}
      <div className="grid gap-6 md:grid-cols-3">
        {steps.map((step, i) => {
          const Icon = step.icon;
          return (
            <Link
              key={step.number}
              href={step.href}
              className={`block border-2 rounded-xl p-6 ${step.lightColor} hover:shadow-md transition-shadow group`}
            >
              <div className="flex items-start gap-4">
                <div
                  className={`${step.color} text-white rounded-lg p-3 flex-shrink-0`}
                >
                  <Icon size={22} />
                </div>
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Step {step.number}
                  </p>
                  <h2 className="font-semibold text-slate-800 text-lg leading-snug">
                    {step.title}
                  </h2>
                  <p className="text-sm text-slate-600">{step.description}</p>
                </div>
              </div>
              <div className="mt-4 flex items-center text-sm font-medium text-slate-700 group-hover:gap-2 transition-all">
                <span>Start here</span>
                <ArrowRight size={14} className="ml-1" />
              </div>
            </Link>
          );
        })}
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800">
        <strong>Legal Disclaimer:</strong> All AI-generated drafts must be reviewed and
        approved by the advocate before filing. This tool does not constitute legal advice.
      </div>
    </div>
  );
}
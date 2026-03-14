import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Legal-CoPilot | DV Petition Drafting",
  description: "AI-assisted domestic violence petition drafting for Maharashtra advocates",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50">
        <nav className="bg-slate-900 text-white px-6 py-4 flex items-center gap-3 shadow-lg">
          <div className="w-8 h-8 bg-amber-600 rounded-full flex items-center justify-center font-bold text-sm">
            ⚖
          </div>
          <span className="font-semibold text-lg tracking-wide">Legal-CoPilot</span>
          <span className="text-slate-400 text-sm ml-1">| DV Petition Drafting</span>
        </nav>
        <main className="container mx-auto px-4 py-8 max-w-5xl">{children}</main>
      </body>
    </html>
  );
}
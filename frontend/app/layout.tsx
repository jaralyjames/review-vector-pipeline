import type { Metadata } from "next";
import "./globals.css";
import "./questions.css";
import "./issues.css";

export const metadata: Metadata = {
  title: "Myntra Review Analyser",
  description: "Explore consolidated insights from Myntra customer reviews.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

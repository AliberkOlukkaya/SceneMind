import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "SceneMind | Video library",
  description: "Local video intelligence and semantic search.",
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

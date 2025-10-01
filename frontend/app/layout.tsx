import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Verba",
  description: "The GoldenRAGtriever",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <link rel="icon" href="platform_icon.ico" />
      <link rel="icon" href="static/platform_icon.ico" />
      <body>{children}</body>
    </html>
  );
}

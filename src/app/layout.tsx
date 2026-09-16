import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Fraunces, Figtree, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const display = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
});

const sans = Figtree({
  variable: "--font-figtree",
  subsets: ["latin"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-ibm",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Drawboard — Giveaway tracker",
  description:
    "Search public social posts for giveaways, lucky draws, and raffle polls across Instagram, Facebook, X, YouTube, Twitch, and more.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${sans.variable} ${mono.variable} h-full`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

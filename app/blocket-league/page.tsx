import type { Metadata } from "next";

import { BlocketLeagueLab } from "@/components/blocket-league/blocket-league-lab";

export const metadata: Metadata = {
  title: "Blocket League — World Model Lab",
  description:
    "A playable two-dimensional world-model experiment: latent prediction, browser inference, and causal activation edits.",
};

export default function BlocketLeaguePage() {
  return <BlocketLeagueLab />;
}

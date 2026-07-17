import type { Metadata } from "next";

import { BlocketLeagueLab } from "@/components/blocket-league/blocket-league-lab";

export const metadata: Metadata = {
  title: "Blocket League — World Model Lab",
  description:
    "A passive pixel transformer learns a tiny physical world, reveals a writable velocity direction, and becomes playable through activation edits.",
};

export default function BlocketLeaguePage() {
  return <BlocketLeagueLab />;
}

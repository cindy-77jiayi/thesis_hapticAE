import type { UIFlow } from "../types";
import { ErrorFlow } from "./ErrorFlow";
import { LoadingFlow } from "./LoadingFlow";
import { MessageFlow } from "./MessageFlow";
import { SuccessFlow } from "./SuccessFlow";

interface FlowRendererProps {
  uiFlow: UIFlow;
  playToken?: number;
  onPlaybackComplete?: () => void;
}

export function FlowRenderer({ uiFlow, playToken = 0, onPlaybackComplete }: FlowRendererProps) {
  if (uiFlow === "success") {
    return <SuccessFlow playToken={playToken} onPlaybackComplete={onPlaybackComplete} />;
  }

  if (uiFlow === "error") {
    return <ErrorFlow playToken={playToken} onPlaybackComplete={onPlaybackComplete} />;
  }

  if (uiFlow === "notification") {
    return <MessageFlow playToken={playToken} onPlaybackComplete={onPlaybackComplete} />;
  }

  return <LoadingFlow playToken={playToken} onPlaybackComplete={onPlaybackComplete} />;
}

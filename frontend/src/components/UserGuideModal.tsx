import { useEffect, useRef } from "react";
import { CircleHelp, X } from "lucide-react";

type GuideKind = "road" | "anpr";

interface UserGuideModalProps {
  kind: GuideKind;
  onClose: () => void;
}

const GUIDE_CONTENT = {
  road: {
    icon: "🛣️",
    title: "How to Use Road Conditions",
    intro: "Follow these steps to scan a road image with the existing AI detector.",
    steps: [
      "Open Road Conditions.",
      "Upload a clear road image.",
      "Click the existing Scan/Analyze button.",
      "Wait while the AI analyzes the image.",
      "Detected conditions such as potholes and waterlogging will be highlighted.",
      "Review the detected condition, confidence score, and available detection details.",
      "If multiple defects are detected, review each detection separately.",
    ],
    tips: [
      "Use a clear image.",
      "Keep the road surface visible.",
      "Avoid extremely dark or blurry images.",
      "Images with clearly visible road defects generally produce better results.",
    ],
    note: "AI predictions may occasionally be incorrect. Important detections should be manually verified.",
  },
  anpr: {
    icon: "🚘",
    title: "How to Use ANPR",
    intro: "Follow these steps to detect and read a vehicle number plate.",
    steps: [
      "Open ANPR.",
      "Upload an image containing a vehicle with a visible number plate.",
      "Click the existing Scan/Detect button.",
      "The AI detects the license plate.",
      "OCR attempts to read the registration number.",
      "Review the detected plate, confidence score, and recognition result.",
    ],
    tips: [
      "Keep the number plate clearly visible.",
      "Use a reasonably close image.",
      "Avoid excessive blur.",
      "Avoid heavily blocked plates.",
      "Avoid strong glare or overexposure.",
    ],
    explanation:
      "The plate may be detected while the registration text is shown as UNKNOWN. This means the detector found the plate, but OCR could not reliably read the characters.",
    note: "Verify important registration numbers manually before taking action.",
  },
} as const;

export function UserGuideModal({ kind, onClose }: UserGuideModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const content = GUIDE_CONTENT[kind];
  const titleId = `${kind}-guide-title`;
  const descriptionId = `${kind}-guide-description`;

  useEffect(() => {
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialogRef.current?.querySelector<HTMLButtonElement>("button")?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }

      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable?.length) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previousFocusRef.current?.focus();
    };
  }, [onClose]);

  return (
    <div
      className="user-guide-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="user-guide-modal panel-elevated animate-fade-in"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <header className="user-guide-header">
          <div className="user-guide-title-group">
            <span aria-hidden="true" className="user-guide-icon">
              {content.icon}
            </span>
            <div>
              <span className="user-guide-kicker">
                <CircleHelp size={14} /> USER GUIDE
              </span>
              <h2 id={titleId}>{content.title}</h2>
            </div>
          </div>
          <button
            type="button"
            className="user-guide-close"
            onClick={onClose}
            aria-label={`Close ${content.title}`}
          >
            <X size={20} />
          </button>
        </header>

        <div className="user-guide-content">
          <p id={descriptionId} className="user-guide-intro">
            {content.intro}
          </p>

          <ol className="user-guide-steps">
            {content.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>

          <section className="user-guide-tips" aria-labelledby={`${kind}-tips-title`}>
            <h3 id={`${kind}-tips-title`}>Tips for better results</h3>
            <ul>
              {content.tips.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </section>

          {"explanation" in content && (
            <p className="user-guide-explanation">{content.explanation}</p>
          )}

          <p className="user-guide-note">{content.note}</p>
        </div>
      </div>
    </div>
  );
}

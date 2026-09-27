import { useEffect, useRef, useState } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ImagePlus,
  Loader2,
  MapPin,
  CircleHelp,
  RefreshCw,
  ScanLine,
  Upload,
  Wrench,
  X,
} from "lucide-react";

import { apiClient } from "../services/api";
import { UserGuideModal } from "../components/UserGuideModal";
import type {
  MaintenanceItem,
  RoadDetectionResult,
  RoadSegment,
  UrbanEvent,
} from "../types";

interface RoadViewProps {
  roadSegments: RoadSegment[];
  events: UrbanEvent[];
  maintenanceQueue: MaintenanceItem[];
  onSelectEvent: (event: UrbanEvent) => void;
}

export const RoadView = ({
  roadSegments,
  events,
  maintenanceQueue,
  onSelectEvent,
}: RoadViewProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<RoadDetectionResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showGuide, setShowGuide] = useState(false);

  const imageRef = useRef<HTMLImageElement | null>(null);

  const [imageSize, setImageSize] = useState({
    width: 1,
    height: 1,
  });

  const defectEvents = events.filter((event) =>
    [
      "pothole",
      "crack",
      "damaged_road",
      "waterlogging",
      "damaged_divider",
      "missing_divider",
      "damaged_zebra",
      "missing_zebra",
      "damaged_sign",
      "missing_sign",
      "road_hazard",
    ].includes(event.event_type),
  );

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const chooseFile = (selectedFile: File | null) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid road image.");
      return;
    }

    if (selectedFile.size > 5 * 1024 * 1024) {
      setError("Image must be smaller than 5 MB.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const clearScan = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setScanning(false);
  };

  const scanRoad = async () => {
    if (!file) {
      setError("Select a road image first.");
      return;
    }

    setScanning(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiClient.detectRoad(file, 0.12);
      setResult(response);
    } catch (err) {
      console.error("Road AI scan failed:", err);

      setError(
        err instanceof Error
          ? err.message
          : "AI scan failed. Check that the backend is running on port 8001.",
      );
    } finally {
      setScanning(false);
    }
  };

  const getConditionColor = (condition: string) => {
    switch (condition) {
      case "good":
        return "#22c55e";
      case "fair":
        return "#eab308";
      case "poor":
        return "#ea580c";
      case "critical":
        return "var(--severity-critical)";
      default:
        return "var(--text-muted)";
    }
  };

  const averageCondition =
    roadSegments.length > 0
      ? Math.round(
          roadSegments.reduce(
            (total, segment) => total + segment.condition_score,
            0,
          ) / roadSegments.length,
        )
      : 0;

  const detectionCount = result?.detection_count ?? 0;
  const isPrototypeWaterlogging = (
    detection: RoadDetectionResult["detections"][number],
  ) =>
    detection.class_name === "waterlogging" &&
    (detection.requires_manual_verification === true ||
      /prototype|heuristic/i.test(detection.detection_method ?? ""));
  const displayedConfidence = (detection: RoadDetectionResult["detections"][number]) =>
    detection.raw_model_confidence ?? detection.confidence;

  return (
    <div
      className="road-page"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "20px",
      }}
    >
      {/* PAGE HEADER */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: "16px",
            flexWrap: "wrap",
          }}
        >
          <div>
            <div
              style={{
                fontSize: "0.7rem",
                letterSpacing: "0.14em",
                fontWeight: 800,
                color: "var(--accent-text)",
                marginBottom: "6px",
              }}
            >
              ROAD INTELLIGENCE
            </div>

            <h2
              className="heading-md"
              style={{ marginBottom: "4px" }}
            >
              Road conditions
            </h2>

            <p
              style={{
                fontSize: "0.875rem",
                color: "var(--text-secondary)",
                margin: 0,
              }}
            >
              Monitor road conditions and scan road images using the trained AI
              model.
            </p>
          </div>

          <div className="guide-actions">
            <button
              type="button"
              className="guide-trigger"
              onClick={() => setShowGuide(true)}
              aria-haspopup="dialog"
            >
              <CircleHelp size={16} />
              How to use
            </button>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "7px",
                padding: "8px 12px",
                borderRadius: "999px",
                border: "1px solid rgba(94, 234, 212, 0.28)",
                background: "rgba(94, 234, 212, 0.08)",
                color: "#78f0d2",
                fontSize: "0.7rem",
                fontWeight: 800,
              }}
            >
              <BrainCircuit size={15} />
              REAL AI CONNECTED
            </div>
          </div>
        </div>

      </div>

      {showGuide && (
        <UserGuideModal kind="road" onClose={() => setShowGuide(false)} />
      )}

      {/* AI ROAD SCANNER */}
      <section
        className="panel"
        style={{
          padding: 0,
          overflow: "hidden",
          borderRadius: "22px",
          border: "1px solid rgba(92, 219, 255, 0.18)",
          background:
            "linear-gradient(145deg, rgba(8,22,31,.96), rgba(8,35,43,.78))",
          boxShadow:
            "0 24px 70px rgba(0,0,0,.28), inset 0 1px rgba(255,255,255,.04)",
        }}
      >
        <div
          style={{
            padding: "20px 22px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "14px",
            flexWrap: "wrap",
            borderBottom: "1px solid rgba(255,255,255,.07)",
          }}
        >
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "0.7rem",
                letterSpacing: "0.14em",
                color: "#7ce8ff",
                fontWeight: 800,
              }}
            >
              <ScanLine size={16} />
              AI ROAD SCANNER
            </div>

            <h3
              style={{
                margin: "7px 0 4px",
                fontSize: "1.3rem",
                color: "var(--text-primary)",
              }}
            >
              Scan a road image
            </h3>

            <p
              style={{
                margin: 0,
                color: "var(--text-secondary)",
                fontSize: "0.82rem",
                maxWidth: "650px",
              }}
            >
              Upload a road photo and the trained model will detect supported
              road defects and show the confidence score.
            </p>
          </div>

          <div
            className="mono"
            style={{
              padding: "8px 12px",
              borderRadius: "999px",
              border: "1px solid rgba(94,234,212,.25)",
              background: "rgba(94,234,212,.08)",
              color: "#78f0d2",
              fontSize: "0.68rem",
              fontWeight: 800,
            }}
          >
            road_defect_best.pt
          </div>
        </div>

        <div
          className="ai-road-grid"
          style={{
            display: "grid",
            gridTemplateColumns:
              "minmax(0, 1.35fr) minmax(280px, 0.65fr)",
            gap: "18px",
            padding: "20px",
          }}
        >
          {/* IMAGE AREA */}
          <div
            style={{
              minHeight: "420px",
              position: "relative",
              borderRadius: "18px",
              overflow: "hidden",
              border: "1px solid rgba(255,255,255,.09)",
              background:
                "radial-gradient(circle at 50% 20%, rgba(45,196,255,.08), transparent 35%), #061117",
            }}
          >
            {!preview ? (
              <label
                style={{
                  minHeight: "420px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "column",
                  padding: "30px",
                  textAlign: "center",
                }}
              >
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  hidden
                  onChange={(event) =>
                    chooseFile(event.target.files?.[0] ?? null)
                  }
                />

                <div
                  style={{
                    width: "74px",
                    height: "74px",
                    display: "grid",
                    placeItems: "center",
                    borderRadius: "22px",
                    marginBottom: "18px",
                    border: "1px solid rgba(104,221,255,.25)",
                    background: "rgba(68,202,255,.08)",
                    boxShadow: "0 0 45px rgba(38,199,255,.12)",
                  }}
                >
                  <ImagePlus size={32} />
                </div>

                <strong
                  style={{
                    fontSize: "1.1rem",
                    color: "var(--text-primary)",
                  }}
                >
                  Upload road image
                </strong>

                <span
                  style={{
                    marginTop: "8px",
                    color: "var(--text-muted)",
                    fontSize: "0.78rem",
                  }}
                >
                  JPG, PNG or WEBP · maximum 5 MB
                </span>
              </label>
            ) : (
              <>
                <img
                  ref={imageRef}
                  src={preview}
                  alt="Road selected for AI analysis"
                  onLoad={(event) => {
                    setImageSize({
                      width: event.currentTarget.naturalWidth,
                      height: event.currentTarget.naturalHeight,
                    });
                  }}
                  style={{
                    display: "block",
                    width: "100%",
                    height: "420px",
                    objectFit: "contain",
                  }}
                />

                {/* REAL MODEL BOUNDING BOXES */}
                {result &&
                  imageRef.current &&
                  result.detections.map((detection, index) => {
                    const image = imageRef.current;

                    if (!image) return null;

                    const naturalRatio =
                      imageSize.width / imageSize.height;

                    const containerRatio =
                      image.clientWidth / image.clientHeight;

                    let renderedWidth = image.clientWidth;
                    let renderedHeight = image.clientHeight;

                    let offsetX = 0;
                    let offsetY = 0;

                    if (naturalRatio > containerRatio) {
                      renderedHeight = renderedWidth / naturalRatio;
                      offsetY =
                        (image.clientHeight - renderedHeight) / 2;
                    } else {
                      renderedWidth = renderedHeight * naturalRatio;
                      offsetX =
                        (image.clientWidth - renderedWidth) / 2;
                    }

                    const left =
                      offsetX +
                      (detection.bbox.x1 / imageSize.width) *
                        renderedWidth;

                    const top =
                      offsetY +
                      (detection.bbox.y1 / imageSize.height) *
                        renderedHeight;

                    const width =
                      ((detection.bbox.x2 - detection.bbox.x1) /
                        imageSize.width) *
                      renderedWidth;

                    const height =
                      ((detection.bbox.y2 - detection.bbox.y1) /
                        imageSize.height) *
                      renderedHeight;

                    return (
                      <div
                        key={`${detection.class_id}-${index}`}
                        style={{
                          position: "absolute",
                          left,
                          top,
                          width,
                          height,
                          border: "2px solid #59f0c7",
                          borderRadius: "6px",
                          boxShadow:
                            "0 0 18px rgba(89,240,199,.45)",
                          pointerEvents: "none",
                        }}
                      >
                        <span
                          style={{
                            position: "absolute",
                            left: "-2px",
                            top: "-29px",
                            whiteSpace: "nowrap",
                            padding: "4px 8px",
                            borderRadius: "6px 6px 6px 0",
                            background: "#59f0c7",
                            color: "#03120f",
                            fontSize: "0.65rem",
                            fontWeight: 900,
                          }}
                        >
                          {isPrototypeWaterlogging(detection) ? `WATERLOGGING · CV score ${Math.round(detection.confidence * 100)}%` : `${detection.class_name.replace(/_/g, " ")} · AI confidence ${Math.round(displayedConfidence(detection) * 100)}%`}
                        </span>
                      </div>
                    );
                  })}

                {/* SCANNING OVERLAY */}
                {scanning && (
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      display: "grid",
                      placeItems: "center",
                      background: "rgba(2,12,17,.72)",
                      backdropFilter: "blur(3px)",
                    }}
                  >
                    <div style={{ textAlign: "center" }}>
                      <Loader2
                        size={38}
                        className="road-ai-spin"
                        color="#70e9ff"
                      />

                      <div
                        style={{
                          marginTop: "12px",
                          fontWeight: 800,
                          color: "white",
                        }}
                      >
                        AI is checking the road...
                      </div>

                      <div
                        style={{
                          marginTop: "5px",
                          fontSize: "0.72rem",
                          color: "#9db2ba",
                        }}
                      >
                        Running trained road-defect model
                      </div>
                    </div>
                  </div>
                )}

                <button
                  type="button"
                  onClick={clearScan}
                  title="Remove image"
                  style={{
                    position: "absolute",
                    right: "12px",
                    top: "12px",
                    width: "38px",
                    height: "38px",
                    display: "grid",
                    placeItems: "center",
                    borderRadius: "12px",
                    border: "1px solid rgba(255,255,255,.15)",
                    background: "rgba(3,13,18,.86)",
                    color: "white",
                    cursor: "pointer",
                  }}
                >
                  <X size={18} />
                </button>
              </>
            )}
          </div>

          {/* SCANNER INFORMATION */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "14px",
            }}
          >
            <div
              style={{
                padding: "17px",
                borderRadius: "16px",
                border: "1px solid rgba(255,255,255,.08)",
                background: "rgba(255,255,255,.025)",
              }}
            >
              <div
                style={{
                  color: "var(--text-muted)",
                  fontSize: "0.68rem",
                  fontWeight: 800,
                  letterSpacing: ".08em",
                }}
              >
                MODEL STATUS
              </div>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  marginTop: "10px",
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  fontSize: "0.82rem",
                }}
              >
                <CheckCircle2 size={17} color="#65e7c5" />
                {result ? 'Model responded' : 'Readiness checked when scanning'}
              </div>
            </div>

            {file && (
              <div
                style={{
                  padding: "17px",
                  borderRadius: "16px",
                  border: "1px solid rgba(255,255,255,.08)",
                  background: "rgba(255,255,255,.025)",
                }}
              >
                <div
                  style={{
                    color: "var(--text-muted)",
                    fontSize: "0.68rem",
                    fontWeight: 800,
                  }}
                >
                  SELECTED IMAGE
                </div>

                <div
                  style={{
                    marginTop: "8px",
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    wordBreak: "break-word",
                    fontSize: "0.8rem",
                  }}
                >
                  {file.name}
                </div>

                <div
                  style={{
                    marginTop: "5px",
                    color: "var(--text-muted)",
                    fontSize: "0.7rem",
                  }}
                >
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            )}

            {/* REAL DETECTION RESULT */}
            {result && (
              <div
                style={{
                  padding: "17px",
                  borderRadius: "16px",
                  border:
                    detectionCount > 0
                      ? "1px solid rgba(101,231,197,.28)"
                      : "1px solid rgba(255,255,255,.08)",
                  background:
                    detectionCount > 0
                      ? "rgba(101,231,197,.055)"
                      : "rgba(255,255,255,.025)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  <strong
                    style={{
                      color: "var(--text-primary)",
                      fontSize: "0.82rem",
                    }}
                  >
                    Scan result
                  </strong>

                  <span
                    style={{
                      fontSize: "0.68rem",
                      fontWeight: 900,
                      color:
                        detectionCount > 0
                          ? "#65e7c5"
                          : "var(--text-muted)",
                    }}
                  >
                    {detectionCount} FOUND
                  </span>
                </div>

                {detectionCount === 0 ? (
                  <p
                    style={{
                      color: "var(--text-muted)",
                      marginBottom: 0,
                      fontSize: "0.75rem",
                      lineHeight: 1.5,
                    }}
                  >
                    No supported road defect passed the current confidence
                    threshold.
                  </p>
                ) : (
                  result.detections.map((detection, index) => (
                    <div
                      key={`${detection.class_id}-${index}`}
                      style={{
                        marginTop: "12px",
                        paddingTop: "12px",
                        borderTop:
                          "1px solid rgba(255,255,255,.07)",
                      }}
                    >
                      <div
                        style={{
                          fontSize: "0.95rem",
                          fontWeight: 800,
                          color: "var(--text-primary)",
                          textTransform: "capitalize",
                        }}
                      >
                        {isPrototypeWaterlogging(detection)
                          ? "WATERLOGGING"
                          : detection.class_name.replace(/_/g, " ")}
                      </div>

                      <div
                        style={{
                          marginTop: "4px",
                          color: "#6ee8cb",
                          fontWeight: 800,
                          fontSize: "0.75rem",
                        }}
                      >
                        {isPrototypeWaterlogging(detection) ? (
                          <>
                            <div>Prototype detection</div>
                            <div>
                              Heuristic score: {Math.round(detection.confidence * 100)}%
                            </div>
                            <div>Manual verification required</div>
                          </>
                        ) : (
                          `${(displayedConfidence(detection) * 100).toFixed(1)}% confidence`
                        )}
                      </div>
                    </div>
                  ))
                )}

                <div
                  style={{
                    marginTop: "12px",
                    paddingTop: "10px",
                    borderTop: "1px solid rgba(255,255,255,.06)",
                    color: "var(--text-muted)",
                    fontSize: "0.65rem",
                  }}
                >
                  Model: {result.model}<br />
                  {result.status || 'Custom trained road detector'}<br />
                  {result.timestamp ? new Date(result.timestamp).toLocaleString() : 'Timestamp unavailable'} · Manual verification required
                </div>
              </div>
            )}

            {error && (
              <div
                style={{
                  padding: "15px",
                  display: "flex",
                  gap: "9px",
                  borderRadius: "15px",
                  border: "1px solid rgba(255,105,105,.25)",
                  background: "rgba(255,80,80,.07)",
                  color: "#ffb4b4",
                  fontSize: "0.75rem",
                }}
              >
                <AlertTriangle
                  size={18}
                  style={{ flexShrink: 0 }}
                />
                <span style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                  <span>{error}</span>
                  <button
                    type="button"
                    onClick={scanRoad}
                    disabled={scanning}
                    style={{
                      border: "1px solid rgba(255,180,180,.35)",
                      borderRadius: "8px",
                      padding: "5px 9px",
                      background: "rgba(255,255,255,.08)",
                      color: "#ffd7d7",
                      cursor: scanning ? "wait" : "pointer",
                      fontWeight: 800,
                      fontSize: "0.7rem",
                    }}
                  >
                    Retry scan
                  </button>
                </span>
              </div>
            )}

            {!file ? (
              <label
                style={{
                  marginTop: "auto",
                  cursor: "pointer",
                  minHeight: "48px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "8px",
                  borderRadius: "13px",
                  border: "1px solid rgba(99,218,255,.25)",
                  background: "rgba(52,192,235,.08)",
                  color: "var(--text-primary)",
                  fontWeight: 800,
                  fontSize: "0.78rem",
                }}
              >
                <Upload size={17} />
                Choose road image

                <input
                  hidden
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) =>
                    chooseFile(event.target.files?.[0] ?? null)
                  }
                />
              </label>
            ) : (
              <button
                type="button"
                onClick={scanRoad}
                disabled={scanning}
                style={{
                  marginTop: "auto",
                  minHeight: "50px",
                  cursor: scanning ? "wait" : "pointer",
                  border: 0,
                  borderRadius: "13px",
                  background:
                    "linear-gradient(135deg,#68e8ff,#63e4c0)",
                  color: "#021216",
                  fontWeight: 900,
                  fontSize: "0.78rem",
                  boxShadow:
                    "0 12px 35px rgba(70,216,235,.18)",
                  opacity: scanning ? 0.7 : 1,
                }}
              >
                {scanning
                  ? "SCANNING ROAD..."
                  : "SCAN WITH REAL AI"}
              </button>
            )}
          </div>
        </div>
      </section>

      {/* SUMMARY CARDS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(190px, 1fr))",
          gap: "14px",
        }}
      >
        <div className="panel" style={{ padding: "16px" }}>
          <div
            style={{
              color: "var(--text-muted)",
              fontSize: "0.7rem",
            }}
          >
            Road sections
          </div>

          <div
            className="mono"
            style={{
              fontSize: "1.6rem",
              fontWeight: 800,
              marginTop: "5px",
            }}
          >
            {roadSegments.length}
          </div>

          <div
            style={{
              color: "var(--text-secondary)",
              fontSize: "0.7rem",
            }}
          >
            Being monitored
          </div>
        </div>

        <div className="panel" style={{ padding: "16px" }}>
          <div
            style={{
              color: "var(--text-muted)",
              fontSize: "0.7rem",
            }}
          >
            Road problems
          </div>

          <div
            className="mono"
            style={{
              fontSize: "1.6rem",
              fontWeight: 800,
              marginTop: "5px",
            }}
          >
            {defectEvents.length}
          </div>

          <div
            style={{
              color: "var(--text-secondary)",
              fontSize: "0.7rem",
            }}
          >
            Current detections
          </div>
        </div>

        <div className="panel" style={{ padding: "16px" }}>
          <div
            style={{
              color: "var(--text-muted)",
              fontSize: "0.7rem",
            }}
          >
            Average condition
          </div>

          <div
            className="mono"
            style={{
              fontSize: "1.6rem",
              fontWeight: 800,
              marginTop: "5px",
            }}
          >
            {averageCondition || "—"}
          </div>

          <div
            style={{
              color: "var(--text-secondary)",
              fontSize: "0.7rem",
            }}
          >
            Road score / 100
          </div>
        </div>

        <div className="panel" style={{ padding: "16px" }}>
          <div
            style={{
              color: "var(--text-muted)",
              fontSize: "0.7rem",
            }}
          >
            Repair queue
          </div>

          <div
            className="mono"
            style={{
              fontSize: "1.6rem",
              fontWeight: 800,
              marginTop: "5px",
            }}
          >
            {maintenanceQueue.length}
          </div>

          <div
            style={{
              color: "var(--text-secondary)",
              fontSize: "0.7rem",
            }}
          >
            Waiting for action
          </div>
        </div>
      </div>

      {/* ROAD CONDITION CARDS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "14px",
        }}
      >
        {roadSegments.map((segment) => (
          <div
            key={segment.id}
            className="panel"
            style={{ padding: "16px" }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "8px",
              }}
            >
              <span
                className="mono"
                style={{
                  fontSize: "0.72rem",
                  color: "var(--text-muted)",
                }}
              >
                {segment.segment_code}
              </span>

              <span
                className="badge"
                style={{
                  backgroundColor: "rgba(255,255,255,.04)",
                  color: getConditionColor(segment.condition),
                  border: `1px solid ${getConditionColor(
                    segment.condition,
                  )}`,
                  fontSize: "0.6875rem",
                  textTransform: "capitalize",
                }}
              >
                {segment.condition}
              </span>
            </div>

            <div
              style={{
                fontWeight: 600,
                color: "var(--text-primary)",
                fontSize: "0.9375rem",
                marginBottom: "10px",
              }}
            >
              {segment.road_name}
            </div>

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                marginBottom: "6px",
              }}
            >
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-muted)",
                }}
              >
                Condition Index:
              </span>

              <span
                className="mono"
                style={{
                  fontWeight: 700,
                  fontSize: "1.25rem",
                  color: getConditionColor(segment.condition),
                }}
              >
                {segment.condition_score}/100
              </span>
            </div>

            <div
              style={{
                width: "100%",
                height: "5px",
                background: "rgba(255,255,255,.06)",
                borderRadius: "3px",
                overflow: "hidden",
                marginBottom: "10px",
              }}
            >
              <div
                style={{
                  width: `${segment.condition_score}%`,
                  height: "100%",
                  background: getConditionColor(segment.condition),
                }}
              />
            </div>

            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.72rem",
                color: "var(--text-muted)",
              }}
            >
              <span>
                Defects:{" "}
                <b style={{ color: "var(--text-primary)" }}>
                  {segment.defect_count}
                </b>
              </span>

              <span>
                Bus passes:{" "}
                <b style={{ color: "var(--text-primary)" }}>
                  {segment.observation_count}
                </b>
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* DETECTED DEFECTS + MAINTENANCE */}
      <div
        className="road-bottom-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "1.2fr 1fr",
          gap: "16px",
        }}
      >
        <div className="panel" style={{ padding: "18px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "14px",
              gap: "10px",
            }}
          >
            <h3
              style={{
                fontSize: "0.9375rem",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <AlertOctagon
                size={16}
                color="var(--accent-text)"
              />

              <span>Detected road problems</span>

            </h3>

            <span
              className="badge badge-neutral"
              style={{ fontSize: "0.6875rem" }}
            >
              {defectEvents.length} Active
            </span>
          </div>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "10px",
            }}
          >
            {defectEvents.length === 0 && (
              <div
                style={{
                  padding: "18px",
                  color: "var(--text-muted)",
                  textAlign: "center",
                }}
              >
                No road problems available.
              </div>
            )}

            {defectEvents.map((event) => (
              <div
                key={event.id}
                onClick={() => onSelectEvent(event)}
                className="clickable-row"
                style={{
                  padding: "12px 14px",
                  borderRadius: "var(--radius-md)",
                  background: "rgba(255,255,255,.02)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  cursor: "pointer",
                }}
              >
                <div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginBottom: "4px",
                    }}
                  >
                    <span
                      style={{
                        fontWeight: 600,
                        color:
                          event.severity === "critical"
                            ? "var(--severity-critical)"
                            : "var(--severity-high)",
                        textTransform: "capitalize",
                        fontSize: "0.8125rem",
                      }}
                    >
                      {event.event_type.replace(/_/g, " ")}
                    </span>

                    <span
                      className="badge badge-low"
                      style={{
                        fontSize: "0.625rem",
                        gap: "4px",
                      }}
                    >
                      <RefreshCw size={10} />
                      {event.observation_count || 1} sightings
                    </span>
                  </div>

                  <div
                    style={{
                      fontSize: "0.78rem",
                      color: "var(--text-secondary)",
                      marginBottom: "4px",
                    }}
                  >
                    {event.description}
                  </div>

                  <div
                    style={{
                      fontSize: "0.6875rem",
                      color: "var(--text-muted)",
                      display: "flex",
                      alignItems: "center",
                      gap: "5px",
                    }}
                  >
                    <MapPin size={11} />
                    {event.latitude.toFixed(4)},{" "}
                    {event.longitude.toFixed(4)}
                  </div>
                </div>

                <div
                  style={{
                    textAlign: "right",
                    flexShrink: 0,
                    marginLeft: "12px",
                  }}
                >
                  <div
                    className="mono"
                    style={{
                      fontWeight: 700,
                      color: "var(--accent-text)",
                      fontSize: "1.125rem",
                    }}
                  >
                    {Math.round(event.confidence * 100)}%
                  </div>

                  <div
                    style={{
                      fontSize: "0.6875rem",
                      color: "var(--text-muted)",
                    }}
                  >
                    Confidence
                  </div>

                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel" style={{ padding: "18px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "14px",
              gap: "10px",
            }}
          >
            <h3
              style={{
                fontSize: "0.9375rem",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <Wrench size={16} color="#22c55e" />
              <span>Repair queue</span>
            </h3>

            <span
              className="badge badge-accent"
              style={{ fontSize: "0.6875rem" }}
            >
              {maintenanceQueue.length} Orders
            </span>
          </div>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "10px",
            }}
          >
            {maintenanceQueue.length === 0 && (
              <div
                style={{
                  padding: "18px",
                  color: "var(--text-muted)",
                  textAlign: "center",
                }}
              >
                No repair jobs waiting.
              </div>
            )}

            {maintenanceQueue.map((item) => (
              <div
                key={item.id}
                style={{
                  padding: "12px 14px",
                  borderRadius: "var(--radius-md)",
                  background: "rgba(255,255,255,.02)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: "10px",
                    marginBottom: "4px",
                  }}
                >
                  <span
                    style={{
                      fontWeight: 600,
                      fontSize: "0.8125rem",
                      color: "var(--text-primary)",
                    }}
                  >
                    {item.title}
                  </span>

                  <span
                    className="mono"
                    style={{
                      fontWeight: 700,
                      fontSize: "0.8125rem",
                      color:
                        item.priority_score > 85
                          ? "var(--severity-critical)"
                          : "var(--severity-high)",
                    }}
                  >
                    {Math.round(item.priority_score)}
                  </span>
                </div>

                <div
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-secondary)",
                    marginBottom: "6px",
                  }}
                >
                  {item.description}
                </div>

                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.6875rem",
                    color: "var(--text-muted)",
                    gap: "10px",
                  }}
                >
                  <span>
                    Status:{" "}
                    <b
                      style={{
                        color: "var(--accent-text)",
                        textTransform: "uppercase",
                      }}
                    >
                      {item.status.replace(/_/g, " ")}
                    </b>
                  </span>

                  <span>
                    Observations:{" "}
                    <b style={{ color: "var(--text-primary)" }}>
                      {item.observation_count}
                    </b>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes roadAiSpin {
          to {
            transform: rotate(360deg);
          }
        }

        .road-ai-spin {
          animation: roadAiSpin 0.8s linear infinite;
        }

        @media (max-width: 900px) {
          .ai-road-grid,
          .road-bottom-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
};

export default RoadView;

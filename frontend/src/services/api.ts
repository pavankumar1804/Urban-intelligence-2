/**
 * Urban Intelligence Platform - API Service & Demo Data Provider
 */
import {
  Bus,
  Route,
  UrbanEvent,
  Alert,
  RoadSegment,
  MaintenanceItem,
  RoadDetectionResult,
} from "../types";

const configuredApiBase = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
// In production, keep browser requests same-origin and let Vercel proxy /api
// to Render. This avoids cross-origin gateway/CORS failures during cold starts.
// Local development can still use VITE_API_BASE_URL or Vite's /api proxy.
const API_BASE = import.meta.env.PROD
  ? "/api"
  : configuredApiBase
    ? configuredApiBase.endsWith("/api")
      ? configuredApiBase
      : `${configuredApiBase}/api`
    : "/api";
const requestedMode = new URLSearchParams(window.location.search).get("mode");
export const DEMO_MODE =
  requestedMode === "demo" ||
  (requestedMode !== "backend" && import.meta.env.VITE_DEMO_MODE !== "false");
let accessToken = sessionStorage.getItem("urban-access-token") || "";
export const isAuthenticated = () => !!accessToken;
export const logout = () => {
  accessToken = "";
  sessionStorage.removeItem("urban-access-token");
  window.dispatchEvent(new Event("urban-auth-change"));
};

type TimedRequestInit = RequestInit & { timeoutMs?: number };

async function fetchWithTimeout(
  input: RequestInfo | URL,
  options: TimedRequestInit = {},
): Promise<Response> {
  const { timeoutMs = 10000, signal, ...requestOptions } = options;
  if (signal) return fetch(input, { ...requestOptions, signal });

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...requestOptions, signal: controller.signal });
  } finally {
    window.clearTimeout(timeoutId);
  }
}
export function switchMode() {
  const url = new URL(window.location.href);
  url.searchParams.set("mode", DEMO_MODE ? "backend" : "demo");
  window.location.assign(url.toString());
}
export async function login(username: string, password: string) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok)
    throw new Error(
      "Sign-in failed. Check your credentials and backend connection.",
    );
  const session = await response.json();
  accessToken = session.access_token;
  sessionStorage.setItem("urban-access-token", accessToken);
  window.dispatchEvent(new Event("urban-auth-change"));
}
export async function request(path: string, options: TimedRequestInit = {}) {
  const headers = new Headers(options.headers);
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  let response: Response;
  try {
    response = await fetchWithTimeout(`${API_BASE}${path}`, {
      ...options,
      headers,
    });
  } catch (error) {
    const name = error instanceof Error ? error.name : "";
    if (name === "TimeoutError" || name === "AbortError") {
      throw new Error("Backend timed out. The free AI service may still be waking up; try once more.");
    }
    throw new Error("Backend is unreachable. Check the deployed API connection.");
  }
  if (response.status === 401 || (response.status === 403 && !accessToken))
    logout();
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.clone().json();
      detail = typeof payload?.detail === "string" ? payload.detail : "";
    } catch {
      // Keep the status-only fallback when the backend returns non-JSON.
    }
    if (response.status === 429) {
      throw new Error("AI service is restarting or busy. Wait about one minute, then scan once.");
    }
    if (response.status === 502 || response.status === 504) {
      throw new Error(`Backend gateway timed out (HTTP ${response.status}). The AI service may be restarting.`);
    }
    if (response.status === 503) {
      throw new Error(`AI model unavailable (HTTP 503)${detail ? `: ${detail}` : "."}`);
    }
    throw new Error(
      `Backend returned HTTP ${response.status}${detail ? `: ${detail}` : "."}`,
    );
  }
  return response;
}

async function waitForDetectionBackend(): Promise<void> {
  // Let one cold-start request finish so the upload can reuse a warm service.
  // This avoids aborting the wake-up and immediately starting another cold
  // connection with a much larger multipart body.
  try {
    await fetchWithTimeout(`${API_BASE}/health`, {
      timeoutMs: 20000,
      cache: "no-store",
    });
  } catch {
    // Continue immediately. The POST below will return the real backend error.
  }
}
export async function getEvidence(id: number) {
  const response = await request(`/events/${id}/evidence`);
  return URL.createObjectURL(await response.blob());
}
export async function getAnalytics<T>(path: string): Promise<T> {
  return (await request(path)).json();
}

// Initial fallback mock data for Hyderabad city
export const MOCK_BUSES: Bus[] = [
  {
    id: 1,
    bus_number: "TS09-3201",
    route_id: 1,
    route_name: "Secunderabad - Charminar",
    status: "active",
    current_latitude: 17.4344,
    current_longitude: 78.5013,
    speed: 38.5,
    heading: 195,
    edge_fps: 21.4,
    active_cameras: 4,
    network_latency_ms: 24,
    passenger_load_pct: 68,
  },
  {
    id: 2,
    bus_number: "TS09-3202",
    route_id: 1,
    route_name: "Secunderabad - Charminar",
    status: "active",
    current_latitude: 17.41,
    current_longitude: 78.488,
    speed: 29.2,
    heading: 210,
    edge_fps: 22.0,
    active_cameras: 4,
    network_latency_ms: 18,
    passenger_load_pct: 54,
  },
  {
    id: 3,
    bus_number: "TS09-3203",
    route_id: 2,
    route_name: "Miyapur - LB Nagar",
    status: "active",
    current_latitude: 17.4735,
    current_longitude: 78.388,
    speed: 44.0,
    heading: 135,
    edge_fps: 20.8,
    active_cameras: 4,
    network_latency_ms: 32,
    passenger_load_pct: 82,
  },
  {
    id: 4,
    bus_number: "TS09-3204",
    route_id: 2,
    route_name: "Miyapur - LB Nagar",
    status: "active",
    current_latitude: 17.45,
    current_longitude: 78.38,
    speed: 22.0,
    heading: 140,
    edge_fps: 23.5,
    active_cameras: 4,
    network_latency_ms: 22,
    passenger_load_pct: 45,
  },
  {
    id: 5,
    bus_number: "TS09-3205",
    route_id: 3,
    route_name: "Kukatpally - Dilsukhnagar",
    status: "active",
    current_latitude: 17.485,
    current_longitude: 78.41,
    speed: 34.0,
    heading: 155,
    edge_fps: 21.9,
    active_cameras: 4,
    network_latency_ms: 29,
    passenger_load_pct: 71,
  },
  {
    id: 7,
    bus_number: "TS09-3207",
    route_id: 4,
    route_name: "ECIL - Mehdipatnam",
    status: "active",
    current_latitude: 17.41,
    current_longitude: 78.468,
    speed: 12.5,
    heading: 245,
    edge_fps: 19.8,
    active_cameras: 4,
    network_latency_ms: 35,
    passenger_load_pct: 89,
  },
  {
    id: 8,
    bus_number: "TS09-3208",
    route_id: 5,
    route_name: "Uppal - Tolichowki",
    status: "active",
    current_latitude: 17.385,
    current_longitude: 78.475,
    speed: 15.0,
    heading: 260,
    edge_fps: 22.4,
    active_cameras: 4,
    network_latency_ms: 27,
    passenger_load_pct: 60,
  },
  {
    id: 11,
    bus_number: "TS09-3211",
    route_id: 2,
    route_name: "Miyapur - LB Nagar",
    status: "active",
    current_latitude: 17.458,
    current_longitude: 78.41,
    speed: 41.2,
    heading: 130,
    edge_fps: 22.7,
    active_cameras: 4,
    network_latency_ms: 25,
    passenger_load_pct: 58,
  },
  {
    id: 12,
    bus_number: "TS09-3212",
    route_id: 1,
    route_name: "Secunderabad - Charminar",
    status: "active",
    current_latitude: 17.44,
    current_longitude: 78.498,
    speed: 31.0,
    heading: 200,
    edge_fps: 21.2,
    active_cameras: 4,
    network_latency_ms: 31,
    passenger_load_pct: 64,
  },
];

export const MOCK_ROUTES: Route[] = [
  {
    id: 1,
    route_number: "R1",
    name: "Secunderabad - Charminar",
    waypoints: [
      [17.4344, 78.5013],
      [17.427, 78.499],
      [17.42, 78.495],
      [17.41, 78.488],
      [17.395, 78.482],
      [17.385, 78.475],
      [17.3616, 78.4747],
    ],
    distance_km: 12.5,
    expected_duration_minutes: 45,
    active_buses: 3,
  },
  {
    id: 2,
    route_number: "R2",
    name: "Miyapur - LB Nagar",
    waypoints: [
      [17.4969, 78.3548],
      [17.485, 78.37],
      [17.4735, 78.388],
      [17.458, 78.41],
      [17.44, 78.435],
      [17.42, 78.455],
      [17.395, 78.478],
      [17.35, 78.51],
    ],
    distance_km: 28.0,
    expected_duration_minutes: 75,
    active_buses: 4,
  },
  {
    id: 3,
    route_number: "R3",
    name: "Kukatpally - Dilsukhnagar",
    waypoints: [
      [17.4948, 78.3996],
      [17.485, 78.41],
      [17.47, 78.425],
      [17.45, 78.44],
      [17.43, 78.455],
      [17.41, 78.47],
      [17.38, 78.5],
    ],
    distance_km: 20.0,
    expected_duration_minutes: 55,
    active_buses: 2,
  },
  {
    id: 4,
    route_number: "R4",
    name: "ECIL - Mehdipatnam",
    waypoints: [
      [17.47, 78.55],
      [17.46, 78.53],
      [17.445, 78.51],
      [17.435, 78.49],
      [17.42, 78.47],
      [17.405, 78.45],
      [17.395, 78.442],
    ],
    distance_km: 22.0,
    expected_duration_minutes: 60,
    active_buses: 3,
  },
  {
    id: 5,
    route_number: "R5",
    name: "Uppal - Tolichowki",
    waypoints: [
      [17.405, 78.559],
      [17.41, 78.535],
      [17.415, 78.51],
      [17.42, 78.485],
      [17.418, 78.46],
      [17.41, 78.435],
      [17.395, 78.418],
    ],
    distance_km: 24.0,
    expected_duration_minutes: 65,
    active_buses: 3,
  },
];

export const MOCK_EVENTS: UrbanEvent[] = [
  {
    id: 101,
    event_id: "EVT_HYD_POT_01",
    event_type: "pothole",
    severity: "high",
    confidence: 0.92,
    latitude: 17.44,
    longitude: 78.498,
    timestamp: new Date(Date.now() - 12 * 60 * 1000).toISOString(),
    bus_id: 12,
    camera_id: 45,
    status: "detected",
    description:
      "Deep road surface crater (0.8m diameter) in central traffic lane on Nampally Road",
    ai_reasoning: [
      "Pothole detected with 92% model confidence via Edge YOLOv8s",
      "High-contrast depression contour with depth shadow profile",
      "Estimated surface disruption area: 0.72 m²",
      "Observed by 2 distinct buses (deduplicated cluster #C104)",
    ],
    is_simulated: true,
    observation_count: 2,
    extra_metadata: { defect_type: "pothole", bus_num: "TS09-3212" },
  },
  {
    id: 102,
    event_id: "EVT_HYD_CONG_02",
    event_type: "congestion",
    severity: "high",
    confidence: 0.89,
    latitude: 17.41,
    longitude: 78.468,
    timestamp: new Date(Date.now() - 8 * 60 * 1000).toISOString(),
    bus_id: 7,
    camera_id: 25,
    status: "acknowledged",
    description:
      "Heavy traffic bottleneck at Mehdipatnam Junction, bus transit lane obstructed",
    ai_reasoning: [
      "Detected 16 vehicles in camera FOV (7 autos, 5 cars, 4 motorcycles)",
      "Average speed 8.4 km/h vs 45 km/h route design speed",
      "Road occupancy density: 88%",
    ],
    is_simulated: true,
    observation_count: 5,
    extra_metadata: {
      vehicle_count: 16,
      avg_speed: 8.4,
      congestion_level: "high",
    },
  },
  {
    id: 103,
    event_id: "EVT_HYD_SAFE_03",
    event_type: "pedestrian_risk",
    severity: "high",
    confidence: 0.91,
    latitude: 17.45,
    longitude: 78.38,
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    bus_id: 4,
    camera_id: 13,
    status: "detected",
    description:
      "Pedestrian group crossing roadway without zebra crossing in Ameerpet School Zone",
    ai_reasoning: [
      "Detected cluster of 4 pedestrians in roadway plane at 7.5m forward proximity",
      "Estimated Time-to-Collision: 2.1s (Bus traveling at 22 km/h)",
      "Active School Zone Geofence hit: Ameerpet Education Hub (200m radius)",
      "Absence of marked zebra crossing",
    ],
    is_simulated: true,
    observation_count: 1,
    extra_metadata: {
      pedestrian_count: 4,
      proximity_m: 7.5,
      school_zone: true,
    },
  },
  {
    id: 104,
    event_id: "EVT_HYD_WATER_04",
    event_type: "waterlogging",
    severity: "critical",
    confidence: 0.95,
    latitude: 17.385,
    longitude: 78.475,
    timestamp: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
    bus_id: 8,
    camera_id: 29,
    status: "assigned",
    description:
      "Severe road water accumulation exceeding 15cm covering bus transit lane",
    ai_reasoning: [
      "Reflective water plane highlights detected across 40% of road surface",
      "Tire submersion and spray optical occlusion verified",
      "Estimated standing water corridor length: 45m",
      "Urgent drainage dispatch required",
    ],
    is_simulated: true,
    observation_count: 4,
    extra_metadata: { defect_type: "waterlogging", severity: "critical" },
  },
  {
    id: 105,
    event_id: "EVT_HYD_INC_05",
    event_type: "hit_and_run",
    severity: "critical",
    confidence: 0.93,
    latitude: 17.458,
    longitude: 78.41,
    timestamp: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
    bus_id: 11,
    camera_id: 41,
    status: "investigating",
    description:
      "Vehicle impact with road divider followed by high-speed departure near Begumpet",
    ai_reasoning: [
      "Vehicle #1084 (White Sedan) lateral impact detected with central road divider",
      "Sudden post-collision acceleration: 18 km/h -> 62 km/h in 3 seconds",
      "Aggressive multi-lane swerve fleeing scene of collision",
      "Automatic ANPR camera trigger dispatched to front and side cameras",
    ],
    is_simulated: true,
    observation_count: 1,
    extra_metadata: {
      vehicle_type: "car",
      speed_kmh: 62.4,
      anpr_triggered: true,
    },
  },
  {
    id: 107,
    event_id: "EVT_HYD_DIV_07",
    event_type: "missing_divider",
    severity: "medium",
    confidence: 0,
    latitude: 17.43,
    longitude: 78.455,
    timestamp: new Date(Date.now() - 31 * 60 * 1000).toISOString(),
    bus_id: 5,
    status: "detected",
    description: "Road divider gap flagged for infrastructure review",
    ai_reasoning: [
      "DEMO RULE: infrastructure-gap scenario; ML confidence not measured",
    ],
    is_simulated: true,
    observation_count: 2,
  },
  {
    id: 108,
    event_id: "EVT_HYD_ZEBRA_08",
    event_type: "missing_zebra",
    severity: "high",
    confidence: 0,
    latitude: 17.395,
    longitude: 78.442,
    timestamp: new Date(Date.now() - 36 * 60 * 1000).toISOString(),
    bus_id: 7,
    status: "confirmed",
    description:
      "Missing marked zebra crossing near high pedestrian footfall zone",
    ai_reasoning: [
      "DEMO RULE: crosswalk deficiency; ML confidence not measured",
    ],
    is_simulated: true,
    observation_count: 4,
  },
  {
    id: 109,
    event_id: "EVT_HYD_SIGN_09",
    event_type: "damaged_sign",
    severity: "medium",
    confidence: 0,
    latitude: 17.42,
    longitude: 78.485,
    timestamp: new Date(Date.now() - 43 * 60 * 1000).toISOString(),
    bus_id: 2,
    status: "detected",
    description: "Damaged traffic signboard flagged for municipal inspection",
    ai_reasoning: [
      "DEMO RULE: signboard deficiency; ML confidence not measured",
    ],
    is_simulated: true,
    observation_count: 1,
  },
  {
    id: 106,
    event_id: "EVT_HYD_ANPR_06",
    event_type: "vehicle_violation",
    severity: "critical",
    confidence: 0.91,
    latitude: 17.4582,
    longitude: 78.4102,
    timestamp: new Date(Date.now() - 3 * 60 * 1000).toISOString(),
    bus_id: 11,
    camera_id: 41,
    status: "confirmed",
    description:
      "ANPR extracted license plate: TS09AB1234 (Telangana Transport Authority format verified)",
    ai_reasoning: [
      "Plate localized with 93% visual confidence",
      "OCR character recognition accuracy: 89%",
      "Format validation: PASS (Indian Motor Vehicle syntax)",
      "Plate marked for law enforcement notification",
    ],
    is_simulated: true,
    observation_count: 1,
    extra_metadata: {
      plate_number: "TS09AB1234",
      ocr_confidence: 0.89,
      syntax_valid: true,
    },
  },
];

export const MOCK_ALERTS: Alert[] = [
  {
    id: 1,
    alert_id: "ALT_001",
    category: "critical",
    title: "Critical Hit-and-Run Incident",
    description:
      "Suspect vehicle TS09AB1234 struck divider and fled scene at Begumpet",
    status: "active",
    created_at: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
  },
  {
    id: 2,
    alert_id: "ALT_002",
    category: "critical",
    title: "Severe Waterlogging Hazard",
    description: "Deep water accumulation (>15cm) on Charminar Bus Corridor",
    status: "active",
    created_at: new Date(Date.now() - 25 * 60 * 1000).toISOString(),
  },
  {
    id: 3,
    alert_id: "ALT_003",
    category: "high",
    title: "School Zone Pedestrian Alert",
    description:
      "Unprotected pedestrian crossing group near Ameerpet Education Hub",
    status: "acknowledged",
    created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
  },
  {
    id: 4,
    alert_id: "ALT_004",
    category: "high",
    title: "Major Pothole Hazard",
    description: "Large road crater verified by 2 buses on Nampally Main Road",
    status: "active",
    created_at: new Date(Date.now() - 12 * 60 * 1000).toISOString(),
  },
  {
    id: 5,
    alert_id: "ALT_005",
    category: "medium",
    title: "Transit Corridor Congestion",
    description:
      "Severe queueing at Mehdipatnam Junction impacting bus schedule",
    status: "investigating",
    created_at: new Date(Date.now() - 8 * 60 * 1000).toISOString(),
  },
];

export const MOCK_ROAD_SEGMENTS: RoadSegment[] = [
  {
    id: 1,
    segment_code: "HYD-RD-01",
    road_name: "Nampally Main Road",
    start_latitude: 17.435,
    start_longitude: 78.49,
    end_latitude: 17.445,
    end_longitude: 78.502,
    condition: "poor",
    condition_score: 42.5,
    defect_count: 5,
    observation_count: 28,
    importance: "high",
  },
  {
    id: 2,
    segment_code: "HYD-RD-02",
    road_name: "Begumpet Airport Arterial",
    start_latitude: 17.45,
    start_longitude: 78.405,
    end_latitude: 17.465,
    end_longitude: 78.42,
    condition: "fair",
    condition_score: 68.0,
    defect_count: 2,
    observation_count: 45,
    importance: "critical",
  },
  {
    id: 3,
    segment_code: "HYD-RD-03",
    road_name: "Charminar Transit Lane",
    start_latitude: 17.358,
    start_longitude: 78.472,
    end_latitude: 17.388,
    end_longitude: 78.48,
    condition: "critical",
    condition_score: 28.0,
    defect_count: 9,
    observation_count: 52,
    importance: "high",
  },
  {
    id: 4,
    segment_code: "HYD-RD-04",
    road_name: "Miyapur Express Corridor",
    start_latitude: 17.48,
    start_longitude: 78.36,
    end_latitude: 17.505,
    end_longitude: 78.385,
    condition: "good",
    condition_score: 91.0,
    defect_count: 0,
    observation_count: 64,
    importance: "normal",
  },
  {
    id: 5,
    segment_code: "HYD-RD-05",
    road_name: "Mehdipatnam Flyover Road",
    start_latitude: 17.4,
    start_longitude: 78.46,
    end_latitude: 17.418,
    end_longitude: 78.475,
    condition: "fair",
    condition_score: 62.0,
    defect_count: 3,
    observation_count: 39,
    importance: "high",
  },
];

export const MOCK_MAINTENANCE: MaintenanceItem[] = [
  {
    id: 1,
    title: "Fill Pothole Crater on Nampally Main Road",
    description:
      "Large road surface depression confirmed by multiple buses. Asphalt patch required.",
    defect_type: "pothole",
    severity: "high",
    priority_score: 88.5,
    observation_count: 3,
    latitude: 17.44,
    longitude: 78.498,
    status: "pending",
    road_segment_id: 1,
  },
  {
    id: 2,
    title: "Emergency Drainage Clearance at Charminar",
    description:
      "Monsoon standing water accumulation exceeding 15cm. Drain unclogging needed.",
    defect_type: "waterlogging",
    severity: "critical",
    priority_score: 95.0,
    observation_count: 12,
    latitude: 17.385,
    longitude: 78.475,
    status: "scheduled",
    road_segment_id: 3,
  },
  {
    id: 3,
    title: "Install Thermoplastic Zebra Crossing at Rethi Bowli",
    description:
      "Heavy pedestrian footfall crossing road near school without marked crosswalk.",
    defect_type: "missing_zebra",
    severity: "medium",
    priority_score: 72.0,
    observation_count: 8,
    latitude: 17.395,
    longitude: 78.442,
    status: "pending",
  },
  {
    id: 4,
    title: "Repair Struck Road Divider at Begumpet",
    description: "Central concrete divider cracked and shifted post-incident.",
    defect_type: "damaged_divider",
    severity: "high",
    priority_score: 82.0,
    observation_count: 5,
    latitude: 17.458,
    longitude: 78.41,
    status: "in_progress",
    road_segment_id: 2,
  },
];

async function getCollection<T>(path: string, demoData: T[]): Promise<T[]> {
  if (DEMO_MODE) return demoData;
  const data: unknown = await (await request(path)).json();
  if (!Array.isArray(data)) throw new Error("Unexpected API response");
  return data as T[];
}

const BACKEND_SCENARIOS = [
  {
    event_type: "pothole",
    severity: "high",
    confidence: 0.92,
    latitude: 17.44,
    longitude: 78.498,
    bus_id: 12,
    description: "Deep road surface crater detected on Nampally Main Road",
  },
  {
    event_type: "pothole",
    severity: "high",
    confidence: 0.95,
    latitude: 17.4402,
    longitude: 78.4981,
    bus_id: 7,
    description: "Corroborating pothole sighting for spatial deduplication",
  },
  {
    event_type: "congestion",
    severity: "high",
    confidence: 0.89,
    latitude: 17.41,
    longitude: 78.468,
    bus_id: 7,
    description: "Traffic congestion bottleneck at Mehdipatnam Junction",
  },
  {
    event_type: "waterlogging",
    severity: "critical",
    confidence: 0.95,
    latitude: 17.385,
    longitude: 78.475,
    bus_id: 8,
    description: "Severe waterlogging covering the bus transit lane",
  },
  {
    event_type: "pedestrian_risk",
    severity: "high",
    confidence: 0.91,
    latitude: 17.45,
    longitude: 78.38,
    bus_id: 4,
    description: "Pedestrian crossing risk in Ameerpet School Zone",
  },
  {
    event_type: "hit_and_run",
    severity: "critical",
    confidence: 0.93,
    latitude: 17.458,
    longitude: 78.41,
    bus_id: 11,
    description: "Hit-and-run incident with vehicle departure near Begumpet",
  },
];

export const apiClient = {
  detectRoad: async (
    file: File,
    confidence = 0.12,
  ): Promise<RoadDetectionResult> => {
    // Wake Render with a lightweight GET before starting CPU-heavy inference.
    // Sending the multipart POST to a sleeping free instance can make the
    // gateway drop the browser connection while inference continues unseen.
    await waitForDetectionBackend();

    const runScan = () => {
      const formData = new FormData();
      formData.append("file", file);
      return request(`/detect/road?confidence=${encodeURIComponent(confidence)}`, {
        method: "POST",
        body: formData,
        // Render may cold-start before CPU inference begins. Bound the wait,
        // but leave enough time for one sequential scan and never auto-retry.
        timeoutMs: 60000,
      });
    };

    const response = await runScan();

    return response.json();
  },
  detectAnpr: async (file: File) => {
    await waitForDetectionBackend();
    const formData = new FormData();
    formData.append("file", file);
    const response = await request("/detect/anpr", {
      method: "POST",
      body: formData,
      timeoutMs: 90000,
    });
    return response.json() as Promise<{
      plate_number: string;
      raw_ocr_text: string;
      plate_detection_confidence: number;
      ocr_confidence: number;
      overall_confidence: number;
      is_format_valid: boolean;
      requires_manual_verification: boolean;
      timestamp: string;
      mode: string;
      warning: string | null;
    }>;
  },
  detectTraffic: async (file: File) => {
    await waitForDetectionBackend();
    const formData = new FormData();
    formData.append("file", file);
    const response = await request("/detect/traffic", {
      method: "POST",
      body: formData,
      timeoutMs: 90000,
    });
    return response.json();
  },
  detectInfrastructure: async (file: File) => {
    await waitForDetectionBackend();
    const formData = new FormData();
    formData.append("file", file);
    const response = await request("/detect/infrastructure", {
      method: "POST",
      body: formData,
      timeoutMs: 90000,
    });
    return response.json();
  },
  detectSafety: async (file: File) => {
    await waitForDetectionBackend();
    const formData = new FormData();
    formData.append("file", file);
    const response = await request("/detect/safety", {
      method: "POST",
      body: formData,
      timeoutMs: 90000,
    });
    return response.json();
  },
  getBuses: async (): Promise<Bus[]> => {
    if (DEMO_MODE) return MOCK_BUSES;
    const rows = await getCollection<any>("/buses/", []);
    return rows
      .filter(
        (b) =>
          Number.isFinite(b.current_latitude) &&
          Number.isFinite(b.current_longitude),
      )
      .map((b) => ({
        ...b,
        speed: b.current_speed ?? 0,
        heading: b.current_heading ?? 0,
        active_cameras: b.cameras?.filter((c: any) => c.status === "online")
          .length,
        last_ping: b.last_telemetry,
      }));
  },
  getRoutes: async (): Promise<Route[]> => {
    if (DEMO_MODE) return MOCK_ROUTES;
    return (await getCollection<any>("/routes/", [])).map((r) => ({
      ...r,
      expected_duration_minutes:
        r.duration_minutes ?? r.expected_duration_minutes,
    }));
  },
  getEvents: async (): Promise<UrbanEvent[]> => {
    if (DEMO_MODE) return MOCK_EVENTS;
    return (await getCollection<any>("/events/", [])).map((e) => ({
      ...e,
      description: e.description || "",
      ai_reasoning: Array.isArray(e.ai_reasoning)
        ? e.ai_reasoning
        : Object.values(e.ai_reasoning || {})
            .flat()
            .filter((v) => typeof v === "string"),
    }));
  },
  getAlerts: () => getCollection("/alerts/", MOCK_ALERTS),
  getRoadSegments: async (): Promise<RoadSegment[]> => {
    if (DEMO_MODE) return MOCK_ROAD_SEGMENTS;
    return (await getCollection<any>("/roads/segments", [])).map((r) => ({
      ...r,
      road_name: r.name,
      segment_code: r.segment_id,
    }));
  },
  getMaintenanceQueue: async (): Promise<MaintenanceItem[]> => {
    if (DEMO_MODE) return MOCK_MAINTENANCE;
    return (await getCollection<any>("/roads/maintenance-queue", [])).map(
      (m) => ({
        ...m,
        observation_count: m.observations,
        latitude: m.lat,
        longitude: m.lng,
      }),
    );
  },
  async triggerDemoScenario(scenarioIndex: number) {
    if (DEMO_MODE) return { status: "simulated_locally", scenarioIndex };

    const scenario = BACKEND_SCENARIOS[scenarioIndex];
    if (!scenario) throw new Error("Unknown demo scenario");
    const response = await request("/events/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...scenario, is_simulated: true }),
      timeoutMs: 10000,
    });
    if (!response.ok)
      throw new Error(`Scenario request failed (${response.status})`);
    return response.json();
  },
  async updateEventStatus(id: number, status: string) {
    if (DEMO_MODE) return { persisted: false, simulated: true };
    return (
      await request(`/events/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      })
    ).json();
  },
  async updateAlertStatus(id: number, status: string) {
    if (DEMO_MODE) return { persisted: false, simulated: true };
    return (
      await request(`/alerts/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      })
    ).json();
  },
};

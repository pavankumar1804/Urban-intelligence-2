import { CityScene } from "../components/CityScene";
import { Reveal, MotionToggle } from "../components/Motion";
import React from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Bus,
  Camera,
  Cpu,
  MapPin,
  ScanLine,
  ShieldCheck,
  Route,
  Layers,
} from "lucide-react";
import { GisMap } from "../components/GisMap";
import { MOCK_BUSES, MOCK_EVENTS, MOCK_ROUTES } from "../services/api";
import "./LandingPage.css";
interface LandingPageProps {
  onOpenDashboard: (tab?: string) => void;
}
export const LandingPage: React.FC<LandingPageProps> = ({
  onOpenDashboard,
}) => (
  <main className="landing-page">
    <div className="ambient-orb ambient-orb-one" aria-hidden="true" />
    <div className="ambient-orb ambient-orb-two" aria-hidden="true" />
    <nav className="site-nav" aria-label="Main navigation">
      <div className="brand">
        <span className="brand-mark">
          <Layers size={22} />
        </span>
        <span>
          Urban Intelligence<small>SAFER STREETS. TOGETHER.</small>
        </span>
      </div>
      <div className="site-nav-links">
        <a
          href="#capabilities"
          onClick={(e) => {
            e.preventDefault();
            document
              .getElementById("capabilities")
              ?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          What you can do
        </a>
        <span>SIH 26124</span>
        <MotionToggle />
        <button
          className="btn btn-primary"
          onClick={() => onOpenDashboard("overview")}
        >
          Open platform <ArrowUpRight size={16} />
        </button>
      </div>
    </nav>
    <section className="landing-hero">
      <div className="landing-copy hero-enter">
        <div className="landing-kicker">
          <span /> SMART INDIA HACKATHON · PROTOTYPE
        </div>
        <h1>
          The city moves.
          <br />
          We connect <em>the dots.</em>
        </h1>
        <p>
          Every bus journey can help make streets safer. See road problems,
          busy traffic, and safety alerts together in one simple place.
        </p>
        <div className="hero-actions">
          <button
            className="btn btn-primary btn-lg"
            onClick={() => onOpenDashboard("overview")}
          >
            Explore your city <ArrowRight size={18} />
          </button>
          <button
            className="btn btn-ghost btn-lg"
            onClick={() => onOpenDashboard("live-map")}
          >
            See the city map <MapPin size={17} />
          </button>
        </div>
        <div className="prototype-note">
          <ShieldCheck size={16} />
          <span>Demo available · AI is still being tested</span>
        </div>
      </div>
      <div className="hero-map-stage">
        <div className="hero-map-orbit" aria-hidden="true" />
        <div className="hero-map">
          <div className="hero-map-title">
            <span>
              <MapPin size={15} /> HYDERABAD NETWORK
            </span>
            <span className="badge badge-demo">Sample data</span>
          </div>
          <GisMap
            buses={MOCK_BUSES}
            routes={MOCK_ROUTES}
            events={MOCK_EVENTS}
            onSelectEvent={() => onOpenDashboard("overview")}
            height="400px"
          />
          <div className="hero-map-footer">
            <span>
              <Bus size={17} /> Cameras on buses
            </span>
            <span>
              <ScanLine size={17} /> Reports with a location
            </span>
          </div>
        </div>
        <div className="map-floating-note">
          <span className="note-glyph">
            <ScanLine size={18} />
          </span>
          <div>
            <strong>Spot it. Find it. Take action.</strong>
            <small>Camera · Map · Review</small>
          </div>
        </div>
      </div>
    </section>
    <Reveal>
      <section className="sensing-flow" aria-label="How the platform works">
        {[
          { icon: Camera, title: "Capture", detail: "Bus cameras + GPS" },
          { icon: Cpu, title: "Understand", detail: "AI checks the video" },
          { icon: MapPin, title: "Locate", detail: "Photos with a location" },
          {
            icon: ShieldCheck,
            title: "Respond",
            detail: "City teams take action",
          },
        ].map((s, i) => (
          <div key={s.title}>
            <span className="flow-number">0{i + 1}</span>
            <s.icon size={23} />
            <section>
              <h2>{s.title}</h2>
              <p>{s.detail}</p>
            </section>
            {i < 3 && <ArrowRight className="flow-arrow" size={17} />}
          </div>
        ))}
      </section>
    </Reveal>
    <Reveal><CityScene compact busCount={MOCK_BUSES.length} eventCount={MOCK_EVENTS.length} onExplore={onOpenDashboard}/></Reveal>
    <Reveal>
      <section className="landing-get-started" aria-labelledby="getting-started-title">
        <div className="getting-started-heading">
          <div className="eyebrow">START HERE</div>
          <h2 id="getting-started-title">New here? Choose what you want to check.</h2>
          <p>
            Pick a tool, upload a clear image, run the scan, and review the AI result.
            Your existing Road Conditions and ANPR scanners are one click away.
          </p>
        </div>

        <div className="getting-started-cards">
          <article className="getting-started-card">
            <div className="getting-started-icon"><ScanLine size={24} /></div>
            <div className="getting-started-card-copy">
              <span className="getting-started-label">ROAD CONDITIONS</span>
              <h3>Check a road image</h3>
              <ol>
                <li>Open Road Conditions.</li>
                <li>Upload a clear road photo.</li>
                <li>Run the existing AI scan.</li>
                <li>Review detected defects and confidence details.</li>
              </ol>
              <p className="getting-started-tip">
                Tip: keep the road surface visible and avoid very dark or blurry images.
              </p>
            </div>
            <button
              className="btn btn-primary"
              onClick={() => onOpenDashboard("roads")}
            >
              Open Road Conditions <ArrowRight size={16} />
            </button>
          </article>

          <article className="getting-started-card">
            <div className="getting-started-icon"><Camera size={24} /></div>
            <div className="getting-started-card-copy">
              <span className="getting-started-label">ANPR</span>
              <h3>Read a vehicle number plate</h3>
              <ol>
                <li>Open Safety &amp; Incidents.</li>
                <li>Find the Live ANPR scanner.</li>
                <li>Upload a clear vehicle or plate photo.</li>
                <li>Review the detected plate and OCR confidence.</li>
              </ol>
              <p className="getting-started-tip">
                Tip: use a close, sharp image with the number plate clearly visible.
              </p>
            </div>
            <button
              className="btn btn-secondary"
              onClick={() => onOpenDashboard("incidents")}
            >
              Open ANPR Scanner <ArrowRight size={16} />
            </button>
          </article>
        </div>

        <p className="getting-started-note">
          AI results can be incorrect. Manually verify important road or registration results before taking action.
        </p>
      </section>
    </Reveal>
    <Reveal>
      <section id="capabilities" className="landing-capabilities">
        <div className="capability-intro">
          <div className="eyebrow">ONE CITY. A CLEARER PICTURE.</div>
          <h2>
            Small details.
            <br />
            A safer city.
          </h2>
          <p>
            A shared workspace for transport teams, road maintenance, and
            incident review.
          </p>
        </div>
        <div className="landing-features">
          {[
            {
              icon: Route,
              title: "Roads & repairs",
              description:
                "Find potholes, flooded roads, and broken signs on the map.",
              tab: "roads",
            },
            {
              icon: Bus,
              title: "Buses & traffic",
              description:
                "See where buses travel and where traffic gets busy.",
              tab: "traffic",
            },
            {
              icon: ShieldCheck,
              title: "Safety & incidents",
              description:
                "Check safety risks and vehicle incidents with photos and details.",
              tab: "incidents",
            },
          ].map((f) => (
            <button
              className="landing-feature"
              key={f.title}
              onClick={() => onOpenDashboard(f.tab)}
            >
              <f.icon size={23} />
              <div>
                <h3>{f.title}</h3>
                <p>{f.description}</p>
              </div>
              <ArrowUpRight size={19} />
            </button>
          ))}
        </div>
      </section>
    </Reveal>
    <footer className="site-footer">
      <span>Urban Intelligence / SIH 26124</span>
      <span>Every journey can make a difference.</span>
    </footer>
  </main>
);

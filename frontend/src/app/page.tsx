import Link from "next/link";
export default function Home() {
  return <main><header><Link className="brand" href="/">SceneMind<span> / workspace</span></Link><span className="status">Local development</span></header><section className="heading"><p className="eyebrow">YOUR WORKSPACE</p><h1>Video library</h1><p>Your videos, organized into moments.</p></section><section className="empty"><span className="film" aria-hidden="true">?</span><h2>A place for every moment.</h2><p>Video upload is coming in the next development phase.</p><a className="button" href="http://localhost:8000/docs">Explore the API ?</a></section><footer>SCENEMIND <span>Private by design. Built for discovery.</span></footer></main>;
}


/** Use browser-managed HTTP credentials; never store the operator password in JS. */
export function apiFetch(input: RequestInfo | URL, init?: RequestInit) {
  return fetch(input, { ...init, credentials: "include" });
}

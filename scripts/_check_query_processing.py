"""Smoke-check query preprocessing against the success-criteria groups.

Prints each group's processed forms so the expansions/variants can be eyeballed,
then asserts that every query in a group shares at least one synonym anchor with
the others — a cheap proxy for "these will retrieve the same article" that runs
without touching Chroma or Postgres.
"""

import sys

sys.path.insert(0, ".")

from backend.app.rag.query_processing import process_query  # noqa: E402

GROUPS = {
    "LMS/Moodle": [
        "How do I log into Moodle?",
        "LMS login",
        "Can't access LMS",
        "moddle login",
        "Learning Management System login",
    ],
    "Email": [
        "Student email",
        "Outlook login",
        "Corporate email",
        "University email",
    ],
    "MFA": [
        "Authenticator setup",
        "MFA",
        "2FA",
        "Microsoft Authenticator",
    ],
    "VAS": [
        "VAS exam",
        "Assessment system",
        "Online exam",
    ],
    "SMOWL": [
        "SMOWL camera",
        "Proctoring software",
        "Exam monitoring",
    ],
    "NL variations": [
        "I forgot my password",
        "can't login",
        "unable to access",
        "where do I sign in",
        "login problem",
    ],
}


def main() -> int:
    failures = []
    for name, queries in GROUPS.items():
        print("=" * 78)
        print(f"GROUP: {name}")
        print("=" * 78)
        anchors = []
        for q in queries:
            p = process_query(q)
            print(f"\n  query      : {q!r}")
            print(f"  normalized : {p.normalized!r}")
            print(f"  lexical    : {p.lexical[:160]!r}")
            print(f"  variants   : {p.variants[1:]}")
            if p.corrections:
                print(f"  corrections: {p.corrections}")
            if p.intents:
                print(f"  intents    : {p.intents}")
            anchors.append(set(p.lexical.lower().split()))

        # Every pair in a group should share vocabulary after expansion.
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                shared = anchors[i] & anchors[j]
                meaningful = {w for w in shared if len(w) > 2}
                if not meaningful:
                    failures.append(
                        f"{name}: {queries[i]!r} vs {queries[j]!r} share no terms"
                    )
        print()

    if failures:
        print("!! NO SHARED VOCABULARY:")
        for f in failures:
            print("  -", f)
        return 1
    print("All groups share vocabulary after expansion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

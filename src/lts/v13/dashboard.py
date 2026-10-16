"""Iteration 13, stage D: World Profitability Dashboard (Basu, Huato, Lara Jauregui, Wasner) via its own buttons.

The dashboard (https://dbasu.shinyapps.io/World-Profitability/, address given in Basu et al. 2025, Review of Political
Economy 37(1)) is a Shiny app; its 'Download Data' buttons create files per selection. This script drives a headless
Chromium (Playwright) through the selections and saves each file with an index row (selection, dashboard file name).
Scope used on 2026-10-16: world and income groups for every data source x currency conversion x aggregation method;
every country (EPWT and WIOD); world by industry (PPP and XR); United States by industry. Industry x every country was
not downloaded, to keep the load on the authors' hosted app small.
Files are registered in MANIFEST afterwards with lts.v13.download.register_local (see data/raw/v13_availability.md).

Usage: python -P -m lts.v13.dashboard <outdir>
"""
import sys, json, glob, os, re, csv, time
from playwright.sync_api import sync_playwright
EXE = (glob.glob('/opt/pw-browsers/chromium*/chrome-linux/chrome') + glob.glob('/opt/pw-browsers/chromium'))[0]
url, outdir = "https://dbasu.shinyapps.io/World-Profitability/", sys.argv[1]
os.makedirs(outdir, exist_ok=True)
idx = open(os.path.join(outdir, "_index.csv"), "a", newline="")
W = csv.writer(idx)
def safe(s): return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
SETJS = """([id, v]) => { const e = document.getElementById(id); if (!e) return 'missing';
  if (e.selectize) { e.selectize.setValue(v); } else { e.value = v; e.dispatchEvent(new Event('change', {bubbles: true})); }
  return 'ok'; }"""
OPTJS = """(id) => { const e = document.getElementById(id); if (!e) return [];
  return e.selectize ? Object.keys(e.selectize.options) : Array.from(e.options).map(o => o.value); }"""
def start(p):
    b = p.chromium.launch(executable_path=EXE, args=["--no-sandbox"])
    ctx = b.new_context(accept_downloads=True, user_agent="LT-research-script/1.0 (noreply@anthropic.com)")
    pg = ctx.new_page(); pg.goto(url, wait_until="domcontentloaded", timeout=120000)
    for _ in range(24):
        pg.wait_for_timeout(5000)
        if pg.locator("#CL_downloadDataGlobal").count(): break
    pg.wait_for_timeout(8000)
    return b, pg
def setv(pg, sid, v, wait=2500):
    r = pg.evaluate(SETJS, [sid, v]); pg.wait_for_timeout(wait); return r
def grab(pg, btn, name, sel):
    out = os.path.join(outdir, name)
    if os.path.exists(out): return
    try:
        with pg.expect_download(timeout=120000) as d:
            pg.locator("#" + btn).first.click()
        dl = d.value; dl.save_as(out)
        W.writerow([name, btn, json.dumps(sel, ensure_ascii=False), dl.suggested_filename, time.strftime("%Y-%m-%dT%H:%M:%S")]); idx.flush()
    except Exception as e:
        W.writerow([name, btn, json.dumps(sel, ensure_ascii=False), f"ERROR {type(e).__name__}: {str(e)[:120]}", time.strftime("%Y-%m-%dT%H:%M:%S")]); idx.flush()
def tab(pg, name, last=False):
    loc = pg.get_by_text(name, exact=True); (loc.last if last else loc.first).click(); pg.wait_for_timeout(4000)
with sync_playwright() as p:
    b, pg = start(p)
    grab(pg, "CL_downloadExplanationFile", "documentation_global-rop.pdf", {})
    # country level: global and groups
    for ds in ("EPWT", "WIOD"):
        for cc in ("PPP", "XR"):
            for agg in ("LimitCountries", "All"):
                tab(pg, "Global")
                setv(pg, "CL_dataSource", ds, 4000); setv(pg, "CL_currencyConversion", cc); setv(pg, "CL_aggregate", agg)
                grab(pg, "CL_downloadDataGlobal", f"cl_global__{ds}_{cc}_{agg}.csv", dict(level="country", tab="Global", source=ds, currency=cc, aggregate=agg))
                tab(pg, "By Group")
                for g in pg.evaluate(OPTJS, "CL_group"):
                    setv(pg, "CL_group", g)
                    grab(pg, "CL_downloadDataGroup", f"cl_group__{ds}_{cc}_{agg}__{safe(g)}.csv", dict(level="country", tab="By Group", source=ds, currency=cc, aggregate=agg, group=g))
        tab(pg, "By Country")
        setv(pg, "CL_currencyConversion", "PPP"); setv(pg, "CL_aggregate", "LimitCountries")
        for c in pg.evaluate(OPTJS, "CL_country"):
            setv(pg, "CL_country", c, 2000)
            grab(pg, "CL_downloadDataCountry", f"cl_country__{ds}__{safe(c)}.csv", dict(level="country", tab="By Country", source=ds, currency="PPP", country=c))
    # industry level
    tab(pg, "Industry-Level")
    inds = pg.evaluate(OPTJS, "IL_industry")
    for cc in ("PPP", "XR"):
        setv(pg, "IL_currencyConversion", cc)
        for ind in inds:
            setv(pg, "IL_industry", ind)
            grab(pg, "IL_downloadDataGlobal", f"il_global__{cc}__{safe(ind)[:80]}.csv", dict(level="industry", tab="Global", currency=cc, industry=ind))
    tab(pg, "By Country", last=True)
    setv(pg, "IL_currencyConversion", "PPP"); setv(pg, "IL_country", "United States")
    for ind in inds:
        setv(pg, "IL_industry", ind)
        grab(pg, "IL_downloadDataCountry", f"il_country__United_States__{safe(ind)[:80]}.csv", dict(level="industry", tab="By Country", currency="PPP", country="United States", industry=ind))
    b.close()
print("done")

import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from anywhere
import csv
from build_profound_csv import rows as prompt_rows

our_prompts = set(p for _, p, _, _, _, _, _, _ in prompt_rows)

with open('profound_citations_raw.csv', encoding='utf-8-sig') as f:
    r = list(csv.DictReader(f))

matches = [row for row in r if row['type'] == 'citation' and row['prompt'] in our_prompts]

seen = set()
deduped = []
for row in matches:
    key = (row['hostname'], row['url'], row['prompt'], row['platform'])
    if key not in seen:
        seen.add(key)
        deduped.append(row)

print(f"raw citation rows for our 26 prompts: {len(matches)}")
print(f"after dedup: {len(deduped)}")

# --- 1. Raw seed export: exactly what the syllabus asks for (Stage 1 output) ---
with open('seed_dataset_raw.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['domain', 'source_url', 'prompt', 'engine'])
    for row in deduped:
        w.writerow([row['hostname'], row['url'], row['prompt'], row['platform']])

# --- 2. Labeling helper: pre-filled suggestion + blank column for manual confirmation ---
DIRECTORY_DOMAINS = {
    'niche.com', 'www.niche.com', 'usnews.com', 'www.usnews.com',
    'petersons.com', 'www.petersons.com', 'collegeboard.org', 'www.collegeboard.org',
    'nces.ed.gov', 'studentsreview.com', 'www.studentsreview.com',
    'unigo.com', 'www.unigo.com', 'collegesimply.com', 'www.collegesimply.com',
    'collegefactual.com', 'www.collegefactual.com', 'princetonreview.com',
    'www.princetonreview.com', 'cappex.com', 'www.cappex.com',
    'bestcolleges.com', 'www.bestcolleges.com', 'collegeconsensus.com',
    'www.collegeconsensus.com', 'collegenavigator.ed.gov',
    # college admissions consulting / ranking-adjacent content sites
    # (primary function is comparing/ranking/advising on schools, same as niche/usnews)
    'blog.collegevine.com', 'www.collegevine.com', 'collegevine.com',
    'www.ivywise.com', 'ivywise.com', 'empowerly.com',
    'www.collegeraptor.com', 'collegeraptor.com',
    'www.varsitytutors.com', 'varsitytutors.com',
    'www.elearners.com', 'elearners.com',
    'www.collegetransitions.com', 'collegetransitions.com',
    'toptieradmissions.com', 'blog.prepscholar.com', 'prepscholar.com',
    'www.collegeadvisor.com', 'collegeadvisor.com',
    'www.ivyscholars.com', 'ivyscholars.com',
    'www.ivycoach.com', 'ivycoach.com',
    'orieladmissions.com', 'meritplaybook.com',
}

# domains that are clearly forums / community discussion, beyond Profound's "Social" tag
FORUM_DOMAINS = {
    'reddit.com', 'www.reddit.com', 'old.reddit.com',
    'quora.com', 'www.quora.com',
    'collegeconfidential.com', 'www.collegeconfidential.com', 'talk.collegeconfidential.com',
    'studentdoctor.net', 'www.studentdoctor.net',
    'www.dcurbanmom.com', 'dcurbanmom.com',
    'www.wallstreetoasis.com', 'wallstreetoasis.com',
}

# additional ranking/directory-style sites found in the actual export that Profound
# had bucketed as generic "Earned Media" but whose primary function is comparing/
# ranking schools (same test as niche.com / usnews.com)
DIRECTORY_DOMAINS_EXTRA = {
    'scholarships360.org', 'www.scholarships360.org',
    'poetsandquantsforundergrads.com', 'www.poetsandquantsforundergrads.com',
    'poetsandquants.com', 'www.poetsandquants.com',
    'research.com', 'www.research.com',
    'eduniversal-ranking.com', 'www.eduniversal-ranking.com',
    'www.collegetuitioncompare.com', 'collegetuitioncompare.com',
    'www.greatvaluecolleges.net', 'greatvaluecolleges.net',
    'www.littleivycolleges.com', 'littleivycolleges.com',
    'www.universityguru.com', 'universityguru.com',
    'www.timeshighereducation.com', 'timeshighereducation.com',
    'guidetobu.com', 'www.guidetobu.com',
    'collegedunia.com', 'www.collegedunia.com',
    'www.collegexpress.com', 'collegexpress.com',
    'myuni.fyi', 'www.myuni.fyi',
}
DIRECTORY_DOMAINS |= DIRECTORY_DOMAINS_EXTRA

# explicit corrections for individual domains Profound miscategorized relative to
# our taxonomy (spot-checked by hand against the taxonomy's "other" examples and
# the actual page content)
MANUAL_OVERRIDES = {
    'en.wikipedia.org': 'other',                 # taxonomy explicitly lists Wikipedia as "other"
    'ajcunet.edu': 'other',                       # Jesuit college trade association, not a single competing school
    'catholic-link.org': 'earned media',          # independent third-party editorial content, not a university
    'uncommoncollege.substack.com': 'earned media',  # written newsletter article, not community discussion
}

PROFOUND_TO_OURS = {
    'Owned': 'BC-owned',
    'Institution': 'competitor institution',
    'Earned Media': 'earned media',
    'Social': 'forum/community',
    'Other': 'other',
    'PR Wire': 'other',
    '': 'other',
}

def suggest_category(hostname, profound_category):
    h = hostname.lower()
    if h in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[h]
    if 'bc.edu' in h:
        return 'BC-owned'
    if h in DIRECTORY_DOMAINS:
        return 'education directory/ranking site'
    if h in FORUM_DOMAINS:
        return 'forum/community'
    return PROFOUND_TO_OURS.get(profound_category, 'other')

from collections import Counter

# domains where the heuristic is confident (exact list match, or clearly bc.edu)
CONFIDENT_DOMAINS = DIRECTORY_DOMAINS | FORUM_DOMAINS | set(MANUAL_OVERRIDES.keys())

cat_counts = Counter()
low_confidence_count = 0

with open('seed_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['domain', 'source_url', 'prompt', 'engine', 'category'])
    for row in deduped:
        h = row['hostname'].lower()
        category = suggest_category(h, row['category'])
        cat_counts[category] += 1
        # flag rows where the label leaned on Profound's coarse category rather than
        # a confident domain match (bc.edu or an explicit list) -- these are the ones
        # most worth a second look, especially anything that landed in "other" or
        # "competitor institution" (a .edu could be a dept./research site, not a rival school)
        confident = ('bc.edu' in h) or (h in CONFIDENT_DOMAINS)
        needs_review = 'yes' if not confident else ''
        if needs_review:
            low_confidence_count += 1
        w.writerow([row['hostname'], row['url'], row['prompt'], row['platform'], category])

print("wrote seed_dataset_raw.csv and seed_dataset.csv")
print("final category counts:", dict(cat_counts))
print(f"rows flagged needs_review (not a confident domain match): {low_confidence_count} of {len(deduped)}")

# cv_pipeline.py
import re
import json
from dateutil import parser
from transformers import pipeline
import pdfplumber

# -------------------------
# 0. Lire le CV depuis un PDF ou TXT
# -------------------------
def load_cv_text(file_path=None):
    if file_path:
        if file_path.lower().endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    # CV par défaut
    return """John Anderson
Backend Software Engineer
Email: john.anderson@email.com
..."""  # CV par défaut ici

# -------------------------
# 1. Extraction Skills
# -------------------------
def extract_skills(cv_text):
    skill_extractor = pipeline(
        "token-classification",
        model="jjzha/jobbert_skill_extraction",
        aggregation_strategy="simple"
    )
    skills_result = skill_extractor(cv_text)
    # Nettoyage : enlever fragments incomplets ou très courts
    skills = set()
    for s in skills_result:
        word = s["word"].strip(" ,.-")
        if len(word) > 2 and not word.startswith("##"):
            skills.add(word)
    return list(skills)

# -------------------------
# 2. Extraction Experience
# -------------------------
def extract_experience(cv_text):
    ner_extractor = pipeline(
        "ner",
        model="Jean-Baptiste/roberta-large-ner-english",
        aggregation_strategy="simple"
    )
    ner_result = ner_extractor(cv_text)
    years = []
    for ent in ner_result:
        if ent["entity_group"] == "DATE":
            try:
                clean_date = re.sub(r"[^\w\s]", "", ent["word"])
                parsed_date = parser.parse(clean_date, fuzzy=True)
                years.append(parsed_date.year)
            except:
                continue
    if years:
        experience = max(years) - min(years)
    else:
        # Tentative via regex sur CV
        matches = re.findall(r'(\d{4})\s*[-–]\s*(\d{4})', cv_text)
        if matches:
            experience = max(int(end) for _, end in matches) - min(int(start) for start, _ in matches)
        else:
            experience = "Not found"
    return experience

# -------------------------
# 3. Zero-Shot Specialty
# -------------------------
def extract_specialty(cv_text):
    classifier = pipeline(
        "zero-shot-classification",
        model="MaVier19/zero-shot_text_classification_fine_tuned"
    )
    labels = ["Backend Development", "Frontend Development", "Data Science", "Cybersecurity", "DevOps", "Administrative"]
    result = classifier(cv_text, labels)
    return result["labels"][0]

# -------------------------
# 4. Scoring RH amélioré
# -------------------------
def scoring_rh(skills, experience):
    score = 0
    importance = {}

    # Poids personnalisés par skill clé
    skill_weights = {
        "java": 20, "spring boot": 20, "docker": 15, "mysql": 15,
        "machine learning": 25, "data": 20, "python": 20,
        "microsoft office": 15, "organization": 15, "communication": 15, "teamwork": 10
    }

    for skill in skills:
        s_lower = skill.lower()
        weight = skill_weights.get(s_lower, 5)  # Poids par défaut
        score += weight
        importance[skill] = round(weight / 20, 2)  # Normalisation

    # Bonus expérience
    if isinstance(experience, int) and experience >= 5:
        score += 20

    return score, importance

# -------------------------
# 5. Run pipeline complet
# -------------------------
def run_pipeline(cv_text):
    skills = extract_skills(cv_text)
    experience = extract_experience(cv_text)
    specialty = extract_specialty(cv_text)
    score, importance = scoring_rh(skills, experience)

    result = {
        "skills": skills,
        "experience_years": experience,
        "specialty": specialty,
        "score": score,
        "importance": importance
    }
    return result

# -------------------------
# 6. Test rapide
# -------------------------
if __name__ == "__main__":
    cv_text = load_cv_text()
    result = run_pipeline(cv_text)
    print(json.dumps(result, indent=4))

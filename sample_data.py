"""
data/sample_data.py
Generates a realistic sample corpus of Ghana 2024 election articles
for demonstration when live scraping is not available.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

OUTLETS = ["MyJoyOnline", "Citinewsroom", "Daily_Graphic", "Ghanaian_Times", "Daily_Guide"]
OUTLET_TYPES = {
    "MyJoyOnline":    "private_online",
    "Citinewsroom":   "private_online",
    "Daily_Graphic":  "state_print",
    "Ghanaian_Times": "state_print",
    "Daily_Guide":    "private_print",
}

ARTICLE_TEMPLATES = {
    "Inflation & Economic Hardship": [
        "Rising fuel prices continue to burden Ghanaian households as the cedi depreciates against major currencies. Traders at Makola Market report sales declining sharply amid the high cost of living.",
        "The Bank of Ghana has raised concerns about inflation which now stands at over 23%, eroding purchasing power for ordinary citizens. Food prices have doubled in the past twelve months.",
        "Petrol prices hit a record high this week, sparking protests in Accra and Kumasi. Transport fares have increased by 30% affecting workers and traders across the country.",
        "Economic hardship continues to define daily life for many Ghanaians as food inflation remains stubbornly high. Markets report lower footfall as consumers cut spending.",
        "The cedi has lost over 40% of its value against the dollar this year. Importers say they are passing costs to consumers, worsening the cost-of-living crisis ahead of December polls.",
        "Ghanaians are spending more on basic necessities but buying less. A survey of 500 households in Accra found 72% cannot afford three square meals daily.",
    ],
    "Illegal Mining (Galamsey)": [
        "The Pra and Offin rivers remain heavily polluted from illegal mining activities despite government pledges to end galamsey. Affected communities demand action.",
        "Environmental groups have released damning footage of destroyed farmland in the Western Region caused by illegal small-scale mining. Cocoa production has plummeted.",
        "Galamsey remains one of the most contentious issues in this election cycle. Both NPP and NDC have been accused of failing to tackle the menace decisively.",
        "The military taskforce against illegal mining has arrested 47 individuals in the Ashanti Region. Critics say enforcement is sporadic and politically motivated.",
        "Water bodies in six regions are now contaminated from mercury used in illegal mining. Health officials warn of long-term kidney and neurological damage for downstream communities.",
        "Farmers in Amansie West say galamsey has destroyed their livelihoods. They plan to vote based on whichever candidate takes the strongest stance against illegal mining.",
    ],
    "Corruption & Governance": [
        "The Special Prosecutor has opened investigations into alleged procurement irregularities worth GHS 800 million at two government ministries.",
        "Opposition NDC has released a dossier accusing the NPP government of judgment debt payments to politically connected firms. The government denies the allegations.",
        "A leaked audit report suggests over GHS 1.2 billion in government funds cannot be accounted for. The Finance Ministry says it is reviewing the findings.",
        "Corruption remains a defining issue for voters according to a pre-election survey. Nearly 41% of respondents say corruption influenced their voting intention.",
        "Civil society organisations are calling for independent audits of all COVID-19 expenditure before the election. They allege over GHS 500 million was misappropriated.",
        "The Auditor-General's report highlights massive irregularities in district assembly spending. Over 200 cases have been referred to the Attorney General.",
    ],
    "NPP Campaign & Manifesto": [
        "Vice President Bawumia unveiled the NPP's 2024 manifesto at a packed rally in Kumasi, promising digital ID for all citizens and an expanded Free SHS programme.",
        "NPP's Alan Kyerematen addressed thousands in Tamale, highlighting infrastructure projects including the Pwalugu dam as evidence of the party's developmental record.",
        "The NPP held its final campaign rally in Accra's Independence Square. Bawumia told supporters the party's record speaks for itself and urged Ghanaians not to reset progress.",
        "NPP parliamentary candidates received campaign materials at the party's headquarters. The party says it is fielding strong candidates in all 276 constituencies.",
        "Bawumia's digital agenda takes centre stage in NPP's campaign messaging. The vice president cited Ghana.gov and mobile money interoperability as flagship achievements.",
        "The NPP says it has created over two million jobs under its watch. Critics say the figures include informal sector workers who would have found work regardless.",
    ],
    "NDC Campaign & Manifesto": [
        "Former President Mahama launched the NDC's 24-hour economy policy in Takoradi, promising round-the-clock industrial activity to create one million jobs.",
        "NDC's running mate Professor Jane Naana Opoku-Agyemang toured Northern Ghana, promising to restore teacher and nursing trainee allowances abolished by the NPP.",
        "John Mahama told a Cape Coast rally that Ghana needs a reset, pointing to the economic hardship and corruption under the current administration.",
        "NDC released its detailed manifesto including GHS 10 billion infrastructure fund targeting rural electrification and roads in underserved communities.",
        "The NDC's big push campaign targets swing constituencies in Brong-Ahafo and the Volta Region. The party says it expects to win 180 parliamentary seats.",
        "Mahama has pledged to reverse all e-levy and other nuisance taxes if elected, a promise that has resonated strongly with small business owners.",
    ],
    "Electoral Process & EC": [
        "The Electoral Commission has confirmed 17.8 million Ghanaians are registered to vote in the December 7 presidential and parliamentary elections.",
        "The EC has deployed 38,000 officials across the country and procured biometric verification devices for all polling stations to prevent multiple voting.",
        "Political parties have been given copies of the final voter register after a month-long exhibition period during which 340,000 new voters were added.",
        "The Electoral Commission chairperson Jean Mensa has assured Ghanaians of a transparent election, rejecting opposition concerns about voter register bloating.",
        "Returning officers for all 276 constituencies have received final collation training. The EC says results will be declared within 48 hours of polls closing.",
        "Domestic and international observers from the African Union, ECOWAS, and EU have begun arriving in Ghana ahead of December 7 elections.",
    ],
    "Education Policy (Free SHS)": [
        "The NPP's Free Senior High School programme remains a campaign battleground. Over 1.3 million students have benefited since its inception in 2017.",
        "NDC has promised to maintain Free SHS while introducing a double-track exit strategy that allows seniors to finish in a shorter time, reducing congestion.",
        "Teachers unions are calling on both parties to address infrastructure deficit in secondary schools. Over 700 schools lack adequate classrooms under the double-track system.",
        "The government says it has spent GHS 4.5 billion on Free SHS since 2017. Opposition critics say quality has declined due to underfunding and overcrowding.",
        "Parents in Northern Ghana say Free SHS has transformed their children's lives. Some communities are seeing secondary school graduates for the first time.",
        "Education researchers warn that Free SHS pass rates have declined. They say quantity without quality improvements risks producing a generation of under-qualified graduates.",
    ],
    "Electoral Security & Violence": [
        "The Ghana Police Service has deployed 55,000 officers nationwide for December 7 elections. Security has been heightened in identified flashpoint constituencies.",
        "CODEO has warned of electoral violence risks in 12 constituencies with history of conflict. It urges political parties to sign the peace pact before election day.",
        "Armed thugs disrupted a campaign rally in Ejura, injuring three persons. Police have arrested four suspects in connection with the incident.",
        "The National Peace Council has called on all stakeholders to maintain calm and avoid inflammatory rhetoric in the final days of campaigning.",
        "Military and police personnel have been jointly deployed to the Northern and Volta regions following reported intimidation of voters by political party operatives.",
        "Both NPP and NDC have signed the 2024 Accra Peace Accord. They pledged to accept results and resolve disputes only through legal means.",
    ],
    "Jobs & Unemployment": [
        "Youth unemployment remains at 13.4% according to the latest Ghana Statistical Service data. Graduate unemployment is significantly higher at nearly 28%.",
        "NABCO graduates are calling on the next government to absorb them permanently into the civil service. Over 100,000 beneficiaries risk losing their stipends.",
        "The NDC's 24-hour economy plan promises to create jobs in manufacturing, hospitality, and logistics by keeping the economy running continuously.",
        "NPP says its YouStart initiative has supported 1,200 young entrepreneurs with seed capital. Critics say the numbers are too small to make a dent in unemployment.",
        "Unemployed graduates in Accra say they cannot find jobs matching their qualifications and are frustrated by government promises that have not materialised.",
        "Traders at Suame Magazine are divided over which party's economic plan will best support their businesses and apprentices who depend on the automotive sector.",
    ],
    "Regional/Parliamentary Races": [
        "The Dome Kwabenya constituency is a key battleground between NPP's Sarah Adwoa Safo and a strong NDC challenger in what analysts call a must-win seat for both parties.",
        "Tamale South remains heavily NDC but NPP is making inroads. Youth turnout is expected to decide the outcome in this urban Northern constituency.",
        "In Kumasi, the NPP is defending nine parliamentary seats. The NDC says it will make unprecedented gains in the Ashanti Region's capital.",
        "The Bolgatanga Central race is too close to call. Both parties have invested heavily in campaigning as the Upper East Region is seen as a bellwether.",
        "Independent candidates are gaining ground in several constituencies where voters are fed up with both major parties. Political analysts see a potential spoiler effect.",
        "The NPP has fielded incumbent MPs in 220 of 276 constituencies. The NDC has replaced several incumbents with younger candidates to energise grassroots support.",
    ],
}

TOPIC_LABELS = list(ARTICLE_TEMPLATES.keys())

TOPIC_WEIGHTS = {
    "MyJoyOnline":    [0.22, 0.14, 0.18, 0.08, 0.12, 0.06, 0.06, 0.05, 0.05, 0.04],
    "Citinewsroom":   [0.20, 0.13, 0.16, 0.09, 0.14, 0.07, 0.05, 0.07, 0.05, 0.04],
    "Daily_Graphic":  [0.12, 0.08, 0.08, 0.14, 0.10, 0.22, 0.10, 0.05, 0.05, 0.06],
    "Ghanaian_Times": [0.10, 0.07, 0.07, 0.15, 0.10, 0.24, 0.10, 0.05, 0.05, 0.07],
    "Daily_Guide":    [0.18, 0.10, 0.20, 0.10, 0.14, 0.08, 0.06, 0.05, 0.05, 0.04],
}

NARROW_TOPICS = {
    "NPP Campaign & Manifesto",
    "NDC Campaign & Manifesto",
    "Electoral Process & EC",
    "Electoral Security & Violence",
    "Regional/Parliamentary Races",
}


def generate_date(month_range=(1, 12)):
    m = random.randint(*month_range)
    d = random.randint(1, 28)
    return datetime(2024, m, d).strftime("%Y-%m-%d")


def generate_corpus(n_per_outlet=60):
    records = []
    for outlet in OUTLETS:
        weights = TOPIC_WEIGHTS[outlet]
        topics_chosen = random.choices(TOPIC_LABELS, weights=weights, k=n_per_outlet)
        for topic in topics_chosen:
            body = random.choice(ARTICLE_TEMPLATES[topic])
            # Slightly vary body text
            filler = random.choice([
                " Analysts say the situation could influence voter behaviour significantly.",
                " Observers note this is likely to feature heavily in final week campaigning.",
                " Political parties have yet to issue a formal response.",
                " Community members expressed mixed opinions when contacted for comment.",
                " This development has sparked debate across social media platforms.",
            ])
            records.append({
                "outlet": outlet,
                "outlet_type": OUTLET_TYPES[outlet],
                "headline": body.split(".")[0][:90],
                "body_text": body + filler,
                "date": generate_date(),
                "lda_topic_label": topic,
                "coverage_type": "Narrow" if topic in NARROW_TOPICS else "Broad",
                "party_frame": (
                    "NPP" if "NPP" in topic else
                    "NDC" if "NDC" in topic else
                    random.choice(["Both", "Neither", "NPP", "NDC"])
                ),
                "word_count": len((body + filler).split()),
                "lda_topic_prob": round(random.uniform(0.45, 0.92), 3),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def get_topic_keywords():
    return {
        "Inflation & Economic Hardship":   ["cedi", "fuel", "prices", "inflation", "economy", "cost", "living", "market", "trader", "depreciate"],
        "Illegal Mining (Galamsey)":        ["galamsey", "mining", "river", "pollution", "forest", "mercury", "cocoa", "farm", "community", "environmental"],
        "Corruption & Governance":          ["corruption", "scandal", "audit", "procurement", "judgment", "debt", "accountability", "investigate", "misappropriate", "irregularity"],
        "NPP Campaign & Manifesto":         ["bawumia", "npp", "manifesto", "rally", "promise", "digital", "infrastructure", "incumbent", "record", "campaign"],
        "NDC Campaign & Manifesto":         ["mahama", "ndc", "reset", "24hour", "economy", "allowance", "promise", "opposition", "manifesto", "job"],
        "Electoral Process & EC":           ["electoral", "commission", "register", "biometric", "polling", "station", "collation", "observer", "transparent", "deploy"],
        "Education Policy (Free SHS)":      ["freeshs", "education", "school", "student", "teacher", "classroom", "curriculum", "double", "track", "quality"],
        "Electoral Security & Violence":    ["security", "police", "military", "violence", "threat", "peace", "constituency", "flashpoint", "intimidation", "deploy"],
        "Jobs & Unemployment":              ["unemployment", "job", "youth", "graduate", "enterprise", "nabco", "youstart", "skill", "create", "sector"],
        "Regional/Parliamentary Races":     ["constituency", "mp", "parliamentary", "seat", "incumbent", "race", "candidate", "regional", "battle", "ward"],
    }


if __name__ == "__main__":
    df = generate_corpus(n_per_outlet=80)
    df.to_csv("data/ghana_election_sample.csv", index=False)
    print(f"Generated {len(df)} articles.")
    print(df["lda_topic_label"].value_counts())

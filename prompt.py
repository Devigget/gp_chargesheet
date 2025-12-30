import requests
import json
import time

# --- CONFIGURATION ---
LLM_API_URL = "https://llama-3b.greenrock-7c76d2df.centralindia.azurecontainerapps.io/v1/chat/completions"
MODEL_ID = "llama3.2-vision:11b"

# --- 1. REAL-WORLD EXAMPLES (Extracted from your PDF) ---
# We organize examples by SECTION so the model focuses only on what matters for that specific part.

EXAMPLES_BRIEF_FACTS = [
    {
        "input": "Case 79/2021. Accused Shobhit Kumar kidnapped minor daughter (16y) from Agacaim to Gujarat, forceful sexual intercourse.",
        "output": """MAY IT PLEASE YOUR HONOUR
In the limits of your Hon'ble Court and within the jurisdiction of Agacaim Police Station that on 11.12.2021 at 10.30 hrs, the accused person mentioned at column No.11 at Sr.No.A-1, kidnapped the complainant's minor daughter age: 16 years from complainant's lawful guardianship and took her to Aslali, Ahmadabad, Gujarat and had forceful sexual intercourse with her, against her wish.
Thereby committed rape.
Thus, the accused person committed an offence punishable u/s 363, 376 IPC and sec. 4 of POCSO Act, 2012.
Hence the charge."""
    },
    {
        "input": "Case 82/2023. Accused Alfred Fernandes trespassed into house at Siolim, entered bedroom, touched private parts without consent.",
        "output": """MAY IT PLEASE YOUR HONOUR
In the limits of your Hon'ble Court and within the Jurisdiction of Anjuna Police Station, that on 15.05.2023 at about 00.30 hrs at Siolim Bardez Goa accused person mentioned at serial No. 11 at Col. No. A-1 criminally trespassed into the house of the complainant and further entered into the complainants bedroom and touched his hand on complainant's vagina and other parts of body and further tried to insert his fingers in her vagina without her consent, thereby committed the rape of the complainant.
Thus the accused lady committed an offence punishable U/s 448, 376 IPC.
HENCE THE CHARGE"""
    }
]

EXAMPLES_PROPERTIES = [
    {
        "input": "Seized: 1. Black long sleeve shirt and grey jeans from accused Shobhit. Sent to FSL.",
        "output": """| Sl. No. | Property Description | Estimated Value (Rs.) | P.S. Property Register No. | From whom/where recovered or seized | Disposal |
|---|---|---|---|---|---|
| 1. | This packed and sealed envelop contains in it one black colour long sleeves shirt having white colour flower design on it and greyish black colour long jeans pant. | - | 01/22 | Attached under arrest cum attachment panchanama at Agacaim PS on 05/01/2022 by PSI Yogesh Gadkar, belongs to the accused Shobhit Kumar. | Sent to FSL Verna for examination. |"""
    },
    {
        "input": "Seized: Bed sheet with blue/white/green design. Seized under scene panchanama 16/05/2023.",
        "output": """| Sl. No. | Property Description | Estimated Value (Rs.) | P.S. Property Register No. | From whom/where recovered or seized | Disposal |
|---|---|---|---|---|---|
| 05. | This greenish colour cloth line sealed envelope contains in it one bed sheet it having blue, white & green colour design on it duly attached under the scene of offence panchanama dated- 16/05/2023 concerned in Anjuna P.S. Cr. No. 82/2023 u/s 448, 376 IPC and marked as "Marked-C". | 250/- | /23 | Attached under scene of offence panchanama dated 16/05/2023 concerned in Anjuna P.S. Cr. No. 82/2023. | Sent at GSFSL Verna for examination and report. |"""
    }
]

EXAMPLES_WITNESSES = [
    {
        "input": "Witnesses: 1. Amida (Complainant), 2. Victim girl (16y), 3. Emran Sab (Panch), 4. Dr. Chetan Karekar (Medical Officer).",
        "output": """| Sr. No. | Name | Fathers/Husband Name | Date/year of birth | Occupation | Address | Type of evidence |
|---|---|---|---|---|---|---|
| 1. | Amida @ Roshan | w/o late Khajasab Budihal | 38 yrs | Labour | c/o Adalpalkar's constructions site Curca, Tiswadi Goa. | Complainant |
| 2. | Miss. Gausiya @ Khushbu | D/o Khajasab Budihal | 16 yrs | Student | --do-- | Victim |
| 3. | Emran Sab Hiremath | Kasim Hiremath | 29 yrs | Business | H.No.1083/2 Near Succor Church, Porvorim Bardez Goa | Panch witness |
| 7. | Dr. Chetan Lavu Karekar | -- | Major | Asst. Lecturer | Dept. of Forensic Medicine & Toxicology, GMC Bambolim. | Medical Officer |"""
    }
]

# --- 2. SECTION DEFINITIONS WITH TARGETED PROMPTS ---
CHARGESHEET_SECTIONS = {
    "1_Initial_Details": {
        "instruction": "Generate the 'Final Form/Report' header table (Form 5.4). Include District, PS, FIR No, Date, and Sections. Use standard format.",
        "examples": "" # Format is standard, fewer examples needed for header usually
    },
    "2_Properties_Seized": {
        "instruction": "Generate the table for 'Details of properties/Articles/Documents recovered/Seized'. Use the exact table headers: Sl. No., Property Description, Estimated Value, P.S. Property Register No., From whom/where recovered, Disposal.",
        "examples": EXAMPLES_PROPERTIES
    },
    "3_Witnesses": {
        "instruction": "Generate the table for 'Particulars of witnesses to be examined'. Columns: Sr. No., Name, Father's/Husband's Name, Age, Occupation, Address, Type of evidence.",
        "examples": EXAMPLES_WITNESSES
    },
    "4_Brief_Facts": {
        "instruction": "Generate 'Brief Facts of the case'. Start strictly with 'MAY IT PLEASE YOUR HONOUR'. Narrate the incident chronologically. End with 'Hence the charge.' or 'HENCE THE CHARGE'.",
        "examples": EXAMPLES_BRIEF_FACTS
    }
}

# --- 3. HELPER FUNCTIONS ---
def askGemini(messages):
    """Call the LLM API."""
    try:
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "max_tokens": 3000,
            "temperature": 0.1  # Low temperature for factual consistency
        }
        response = requests.post(LLM_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"ERROR: {e}")
        return ""

def format_few_shot(examples):
    """Converts the list of dicts into a string for the prompt."""
    if not examples: return ""
    text = "Here are examples of the expected output format:\n"
    for ex in examples:
        text += f"\n[EXAMPLE INPUT]:\n{ex['input']}\n[EXAMPLE OUTPUT]:\n{ex['output']}\n"
    return text

# --- 4. NEW CASE DATA (USER INPUT) ---
# Update this block with your new case details
NEW_CASE_FACTS = """
Case No. 789/2025. Station: Mapusa PS. 
Accused: Geeta Devi (Age 45, R/o Mapusa). 
Sections: 376 IPC (Abetment/Involvement). 
Incident: On 05/11/2025, Accused Geeta Devi allowed a male accomplice to enter the house 
where the minor victim was sleeping. She stood guard outside. 
Property Seized: One mobile phone Samsung M31 used to call the accomplice, seized from Geeta Devi.
Witnesses: 1. Victim (Minor), 2. Mother of Victim (Complainant), 3. Panch Witness Ramesh.
"""

# --- 5. MAIN GENERATION LOOP ---
final_document = ""

print(f"Starting Generation for Case 789/2025...\n")

for section_name, section_data in CHARGESHEET_SECTIONS.items():
    print(f"Generating {section_name}...")
    
    # Prepare the Prompt
    few_shot_text = format_few_shot(section_data["examples"])
    
    system_prompt = f"""You are a Legal Drafting Assistant for Indian Criminal Law. 
    Task: Draft the section '{section_name}' for a Police Final Report (Chargesheet).
    
    GUIDELINES:
    {section_data['instruction']}
    
    {few_shot_text}
    """
    
    user_prompt = f"Using the following facts, generate the section '{section_name}':\n\n{NEW_CASE_FACTS}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Call LLM
    content = askGemini(messages)
    
    if content:
        final_document += f"\n\n{'='*30}\nSECTION: {section_name}\n{'='*30}\n{content}"
        print(f" -> {section_name} Completed.")
    else:
        print(f" -> {section_name} Failed.")
    
    time.sleep(1) # Polite delay

# --- 6. SAVE OUTPUT ---
with open("Generated_Chargesheet.txt", "w", encoding="utf-8") as f:
    f.write(final_document)

print("\nProcessing Complete. Output saved to 'Generated_Chargesheet.txt'.")

import sys
import pandas as pd
from langchain.prompts import ChatPromptTemplate
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieval

# Định nghĩa template với các placeholder
template = """
{p1}
{p6}
{p2}
{p3}
{p4}
{p5}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def behavior(row):
    id = row["ID"]
    behavior_path = "behavior/behavior.csv"
    behavior_df = pd.read_csv(behavior_path)
    behavior_row = behavior_df[behavior_df["ID"] == id]
    if behavior_row.empty:
        return "No transportation behavior information."
    else:
        return behavior_row["response"].iloc[0]

def prompt(row):
    # p1: Task description
    p1 = "You are a transportation behavior expert that predicts trip mode (either 'Public transports (train, bus, tram, etc.)', 'Private modes (car, motorbike, etc.)', 'Soft modes (bike, walk, etc.)'). Based on the provided trip details, their previous trip choices, transportation behavior and similar past trips, what is the most likely trip mode? Only output one of: [Public transports, Private modes, Soft modes]. Note the '...' places are where information is missing. "
    
    #p6: Processing logic
    p6 = "Let's understand the problem and solve it step by step. "
    # p6 += "Step 1. Analyze the causal relationship or tendency between each feature and task description based on general knowledge and common sense within a short sentence. "
    # p6 += "Step 2. Based on the above examples, previous choices and Step 1's results, infer 12 conditions corresponding to 12 variables when choosing mode of transport. The conditions must be clear and make sense. "
    # p6 += "Step 3. Based on the above Step 2's conditions, trip details and transportation behavior infer the most likely trip mode that the person will choose. "
    
    #Note the '...' places are where information is missing.   
    
    #p2: Features description
    p2 = "Features:"
    p2 += "- TimePT: Total duration of the travel performed in public transport (in minutes)\n"
    p2 += "- TimeCar: Total duration of the travel performed using the car (in minutes)\n"
    p2 += "- MarginalCostPT: Total cost in public transport taking into account the possible discounts\n"
    p2 += "- CostCarCHF: Total cost of a travel performed using the car (in Swiss francs)\n"
    p2 += "- NbTransf: The total number of transfers performed for all trips of the loop, using public transport\n"
    p2 += "- Distance km: Total distance (in kilometers)\n"
    p2 += "- Trip purpose: Purpose of the trip\n"
    p2 += "- Age: Age of the individual\n"
    p2 += "- Gender: Gender of the individual\n"
    p2 += "- StudyLevel: Study level of the individual\n"
    p2 += "- Income: Monthly income of the household (in Swiss francs)\n"
    p2 += "- NbCar: Number of cars in the household\n"
    p2 += "- NbMoto: Number of motorbikes in the household\n" 
    p2 += "- NbBicy: Number of bicycles in the household\n"
    p2 += "- CarAvail: Availability of a car in the household for going out"
    
    # p3: Input information
    p3 = "Trip details: " + row["INFOR"]
    
    #p5: Transportation behavior
    p4 = "Transportation behavior: " + behavior(row)
    
    # p4: Retrieval information
    situations, examples = retrieval(row["INFOR"], row["ID"])
    
    formatted_results = "The person's previous choices: "
    if situations:
        formatted_results += "".join(situations)
    else:
        formatted_results += "No previous choices information. "
    formatted_results += "The similar trips from others: "
    if examples:
        formatted_results += "".join(examples)
    else:
        formatted_results += "No similar trips information."
    
    p5 = formatted_results
    

    final_prompt = prompt_template.invoke({
        "p1": p1,
        "p6": p6,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5
    })
    
    return final_prompt
    
# test prompt
if __name__ == "__main__":
    df = pd.read_csv("data/test.csv")
    row = df.iloc[19]
    prompt_value = prompt(row)
    result = str(prompt_value)  # convert to string

    print(result)
    print(row["ID"])
    
    
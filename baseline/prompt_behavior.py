from asyncore import poll3
from poplib import POP3_PORT
import sys
import pandas as pd
from langchain.prompts import ChatPromptTemplate
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Định nghĩa template với các placeholder
template = """
{p1}
{p2}
{p3}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def prompt(row):
    # p1: Task description
    p1 = f"You are a transportation behavior expert that thinking about the person's transportation behavior. Based on attitude questions about mobility you have 2 tasks.\n"
    p1 += f"The first task is to infer 3 conditions that describe the person's transportation behavior when choosing 'Public transports (train, bus, tram, etc.)', 'Private modes (car, motorbike, etc.)', 'Soft modes (bike, walk, etc.)'.\n"
    p1 += f"The second task is to list the key exceptions and specific considerations when deciding on the most suitable means of transportation for a person. These factors should include elements they strongly agree and those they strongly disagree. But must not use any internal codes or labels.\n"
    p1 += f"The output is a paragraph containing information about three conditions in the format: 'That person uses Public transports if: [condition], Use Private modes if: [condition], Use Soft modes if: [condition]. [Exception or consideration].'. The paragraph must be clear and in the correct format."
    
     #p2: question description
    p2 = "There are 5 levels of responses: 1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree, 6 = not applicable. (note: -1 = missing value). "
    p2 += "Questions:\n"
    p2 += "Mobil01: I use the time of my trip in a productive way.\n"
    p2 += "Mobil02: Being stuck in traffic bores me.\n"
    p2 += "Mobil03: I reconsider frequently my mode choice.\n"
    p2 += "Mobil04: I use my current mean of transport mode because I have no alternative.\n"
    p2 += "Mobil05: In general, for my activities, I always have a usual mean of transport.\n"
    p2 += "Mobil06: I do not feel comfortable when I travel close to people I do not know.\n"
    p2 += "Mobil07: Taking the bus helps making the city more comfortable and welcoming.\n"
    p2 += "Mobil08: With my car I can go wherever and whenever.\n"
    p2 += "Mobil09: When I take the car I know I will be on time.\n"
    p2 += "Mobil10: I do not like looking for a parking place.\n"
    p2 += "Mobil11: I do not like changing the mean of transport when I am traveling.\n"
    p2 += "Mobil12: If I use public transportation I have to cancel certain activities I would have done if I had taken the car.\n"
    p2 += "Mobil13: CarPostal bus schedules are sometimes difficult to understand.\n"
    p2 += "Mobil14: I know very well which bus/train I have to take to go where I want to.\n"
    p2 += "Mobil15: I know by heart the schedules of the public transports I regularly use.\n"
    p2 += "Mobil16: I have always used public transports all my life.\n"
    p2 += "Mobil17: I know some drivers of the public transports that I use."
    
    
    # p3: Input information
    p3 = "Responses to the questions: " 
    p3 += "Mobil01: " + str(row["Mobil03"])
    p3 += ". Mobil02: " + str(row["Mobil04"])
    p3 += ". Mobil03: " + str(row["Mobil05"])
    p3 += ". Mobil04: " + str(row["Mobil06"])
    p3 += ". Mobil05: " + str(row["Mobil07"])
    p3 += ". Mobil06: " + str(row["Mobil08"])
    p3 += ". Mobil07: " + str(row["Mobil09"])
    p3 += ". Mobil08: " + str(row["Mobil13"])
    p3 += ". Mobil09: " + str(row["Mobil14"])
    p3 += ". Mobil10: " + str(row["Mobil15"])
    p3 += ". Mobil11: " + str(row["Mobil16"])
    p3 += ". Mobil12: " + str(row["Mobil17"])
    p3 += ". Mobil13: " + str(row["Mobil18"])
    p3 += ". Mobil14: " + str(row["Mobil19"])
    p3 += ". Mobil15: " + str(row["Mobil20"])
    p3 += ". Mobil16: " + str(row["Mobil24"])
    p3 += ". Mobil17: " + str(row["Mobil26"])
    

    
    final_prompt = prompt_template.invoke({
        "p1": p1,
        "p2": p2,
        "p3": p3
    })
    
    return final_prompt
    
# test prompt
if __name__ == "__main__":
    df = pd.read_csv("data/behavior.csv")
    row = df.iloc[17]
    prompt_value = prompt(row)
    result = str(prompt_value)  # convert to string

    print(result)
    
    
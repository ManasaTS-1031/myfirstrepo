import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
context="""
Title: About Chronic Diseases
 
URL Source: http://www.cdc.gov/chronic-disease/about/index.html
 
Markdown Content:
For Everyone
 
[May 14, 2026, visit link for details.](http://www.cdc.gov/chronic-disease/about/index.html#content-sources)
 
*   Chronic diseases are the leading cause of illness, disability, and death in America.
*   Most chronic diseases are caused by a short list of risk factors: smoking, poor nutrition, physical inactivity, and excessive alcohol use.
*   Some groups are more affected than others because of factors that limit their ability to make healthy choices.
 
![Image 1: Person getting his blood pressure measured](http://www.cdc.gov/chronic-disease/media/files/2025/07/HEhighbloodpressure-reduced.webp)
 
[](http://www.cdc.gov/chronic-disease/about/index.html "Chronic diseases in America")
## Chronic diseases in America
 
### Definition
 
Chronic diseases are defined broadly as conditions that last 1 year or more and require ongoing medical attention or limit activities of daily living or both.
 
Chronic diseases such as [heart disease](http://www.cdc.gov/heart-disease/about/index.html), [cancer](http://www.cdc.gov/cancer/index.html), and [diabetes](http://www.cdc.gov/diabetes/about/index.html) are the leading causes of death and disability in the United States. They are also leading drivers of the nation's $5.3 trillion in annual [health care costs](http://www.cdc.gov/chronic-disease/data-research/facts-stats/index.html).[1](http://www.cdc.gov/chronic-disease/about/index.html#cdcreference_1)[2](http://www.cdc.gov/chronic-disease/about/index.html#cdcreference_2)[3](http://www.cdc.gov/chronic-disease/about/index.html#cdcreference_3)
 
Three in four American adults have at least one chronic condition, and over half have two or more chronic conditions.[4](http://www.cdc.gov/chronic-disease/about/index.html#cdcreference_4)
 
*   Among adults ages 65 and older, more than 90% **have at least one** chronic condition.
*   Among midlife adults ages 35–64, more than 75% **have at least one** condition.
*   Among younger adults ages 18–34, 60% have **at least one** condition.
 
Many preventable chronic diseases are caused by a short list of risk behaviors: smoking, poor nutrition, physical inactivity, and excessive alcohol use.
 
[](http://www.cdc.gov/chronic-disease/about/index.html "Risk factors")
## Risk factors
 
### Smoking
 
[Cigarette smoking](http://www.cdc.gov/tobacco/about/index.html) causes more than 480,000 deaths each year in the United States, and over 16 million Americans are living with a disease caused by smoking. Smoking causes cancer, heart disease, stroke, lung diseases, diabetes, and chronic obstructive pulmonary disease (COPD), which includes emphysema and chronic bronchitis.
 
### Poor nutrition and physical inactivity
 
[Poor nutrition](http://www.cdc.gov/nutrition/php/about/index.html) and [physical inactivity](http://www.cdc.gov/physical-activity/php/about/index.html) are significant risk factors for obesity and other chronic diseases, such as type 2 diabetes, heart disease, stroke, certain cancers, and depression.
 
### Excessive alcohol use
 
Over time, [excessive alcohol use](http://www.cdc.gov/alcohol/about-alcohol-use/index.html) can lead to serious problems, including alcohol use disorder and problems with learning, memory, and mental health. Chronic health conditions linked to excessive alcohol use include high blood pressure, heart disease, stroke, liver disease, and some kinds of cancer.
 
[](http://www.cdc.gov/chronic-disease/about/index.html "Who is at risk")
## Who is at risk
 
Some groups are at higher risk of chronic diseases because of conditions where they are born, live, work, and age. These nonmedical factors, called social determinants of health, can be positive or negative. When they are negative, they limit the opportunities to make healthy choices and get good medical care.
 
For example, some communities lack safe spaces like parks for people to be active, or grocery stores that sell fresh fruits and vegetables. In some rural areas, it's hard to get medical care because of doctor shortages, hospital closures, or long distances to care. This makes it challenging to get preventive screenings or specialist follow-up care.
 
[](http://www.cdc.gov/chronic-disease/about/index.html "What CDC is doing")
## What CDC is doing
 
CDC's National Center for Chronic Disease Prevention and Health Promotion supports state, local, tribal, and territorial public health organizations to reduce chronic disease risk factors. [Funded programs](http://www.cdc.gov/health-equity-chronic-disease/nccdphps-programs-to-address-social-determinants-of-health/index.html) focus on addressing the social determinants of health, so that everyone can have the same opportunity to live their healthiest life.
 
 May 14, 2026 
 
[Sources and Page Info](http://www.cdc.gov/chronic-disease/about/index.html#content-sources)[Print](http://www.cdc.gov/chronic-disease/about/index.html#print)[Share](http://www.cdc.gov/chronic-disease/about/index.html#share)
 
Content Source:
 
[National Center for Chronic Disease Prevention and Health Promotion](http://www.cdc.gov/nccdphp/index.html)
 
About this page
 
Published: May 15, 2024
 
Updated: May 14, 2026
 
This page was last updated on this date. Updates may include minor edits, image changes, or other modifications to page content.
 
Reviewed: May 14, 2026
 
The information on this page was last reviewed by subject matter experts to ensure accuracy.
 
References
 
1.   National health expenditure data: historical. Center for Medicare & Medicaid Services. Updated December 18, 2024. Accessed March 4, 2025. [https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/historical](https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/historical)
2.   Buttorff C, Ruder T, Bauman M. _Multiple Chronic Conditions in the United States_ _._ Rand Corp.; 2017.
3.   Leading causes of death. Centers for Disease Control and Prevention. Updated January 23, 2023. Accessed November 7, 2023. [https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm](https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm)
4.   Watson KB, Wiltz JL, Nhim K, Kaufmann RB, Thomas CW, Greenlund KJ. Trends in Multiple Chronic Conditions Among US Adults, By Life Stage, Behavioral Risk Factor Surveillance System, 2013–2023. Prev Chronic Dis 2025;22:240539. DOI: [http://dx.doi.org/10.5888/pcd22.240539](http://dx.doi.org/10.5888/pcd22.240539)
 
Sources
 
*   Chronic disease definition: US Department of Health and Human Services. Multiple Chronic Conditions: A Strategic Framework; 2010.

"""
load_dotenv()

key = os.getenv("GEMINI_KEY")
question="can you tell about chronic disease?"
client = genai.Client(api_key=key)
system_prompt=f"""
You are a medical information assistant answer from the provided context.
if no context is given or you didnt find any anwer tell no context provided.


###  Context

{context}

### User Question

{question}

"""
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[system_prompt, question],
    config=types.GenerateContentConfig(
        temperature=0.0
    )
)

res = response.text.strip()

print(res)
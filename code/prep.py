import pandas as pd
import numpy as np

def prep(data: pd.DataFrame):
    
    data = data.copy()

    keep = [
        "Hobbyist",
        "Age",
        "Wage",
        "DatabaseWorkedWith",
        "DevType",
        "EdLevel",
        "Employment",
        "Ethnicity",
        "Gender",
        "LanguageWorkedWith",
        "MiscTechWorkedWith",
        "NEWCollabToolsWorkedWith",
        "NEWDevOps",
        "NEWLearn",
        "NEWOvertime",
        "OpSys",
        "OrgSize",
        "PlatformWorkedWith",
        "PurchaseWhat",
        "Sexuality",
        "Trans",
        "UndergradMajor",
        "WebframeWorkedWith",
        "YearsCode",
        "YearsCodePro"
    ]

    groups = {
        "Demographics": ["Age", "Ethnicity", "Gender", "Sexuality", "Trans"],
        "Education": ["EdLevel", "UndergradMajor"],
        "Experience": ["DatabaseWorkedWith", "LanguageWorkedWith", "MiscTechWorkedWith", "NEWCollabToolsWorkedWith", "OpSys", 
                    "PlatformWorkedWith", "PurchaseWhat", "WebframeWorkedWith", "YearsCode", "YearsCodePro"],
        "Job": ["DevType", "Employment", "NEWOvertime"],
        "Personal_Professional": ["Hobbyist", "NEWLearn"], 
        "Company": ["NEWDevOps", "OrgSize"]
    }

    numeric = ["Age", "Wage", "YearsCode", "YearsCodePro"]
    categorical = keep.copy()

    for item in keep:
        if item in numeric:
            categorical.remove(item)

    base = {
        "Hobbyist": "No",
        "DatabaseWorkedWith": "MySQL",
        "DevType": "Developer, full-stack",
        "EdLevel": "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
        "Employment": "Employed full-time",
        "Ethnicity": "White or of European descent",
        "Gender": "Man",
        "LanguageWorkedWith": "JavaScript",
        "MiscTechWorkedWith": "Node.js",
        "NEWCollabToolsWorkedWith": "Jira",
        "NEWDevOps": "No",
        "NEWLearn": "Once every few years",
        "NEWOvertime": "Never",
        "OpSys": "Windows",
        "OrgSize": "2 to 9 employees",
        "PlatformWorkedWith": "Windows",
        "PurchaseWhat": "I have little or no influence",
        "Sexuality": "Straight / Heterosexual",
        "Trans": "No",
        "UndergradMajor": "Computer science, computer engineering, or software engineering",
        "WebframeWorkedWith": "React.js",
    }

    # For this analysis, we are focused on:
    #   U.S. developers
    #   Professional developers
    #   At least 18 years of age
    data = data[data["Country"] == "United States"]
    data = data[data["MainBranch"] == "I am a developer by profession"]
    data = data[data["Age"] >= 18]

    #
    data = data.dropna(subset=["YearsCode", "YearsCodePro"])

    for col in ["YearsCode", "YearsCodePro"]:
        data[col].loc[data[col] == "Less than 1 year"] = 0.5
        data[col].loc[data[col] == "More than 50 years"] = 51
        data[col] = data[col].astype("float")

    # As we are dealing with self-reported income data, it is possible that some folks have lied about their income
    # with absurdly high or low values. In order to filter for this I remove some high and low values outside the 
    # typical earnings range for software developers. This is admittedly a coarse approach, but likely not far from
    # the optimal approach.
    data = data[(data["ConvertedComp"] <= 500000) & (data["ConvertedComp"] > 5000)]

    # We then transform the annual income values into houry earnings by dividing by 50 (which Stack Overflow assumes
    # as the number of working weeks) and then dividing by the self-reported number of work week hours
    data["Wage"] = np.log(data["ConvertedComp"] / 50 / data["WorkWeekHrs"])

    # Given income is a major focus of the analysis, we drop the small number of respondents with missing income
    #
    # We then replace any other missing values with "no_answer", which we will explicitly control for later
    print(f'Removing {data["Wage"].isna().sum()} respondents with missing income')
    data = data.dropna(subset=["Wage"])
    data = data.fillna("no_answer")

    # Given the focus on black developers and the very small sample proportion, coding all developers who identify as at least
    # partially black as black. All other multiracial individuals coded as multiracial
    #
    # Anecdotally, many developers who are multiracial blacks self-indentify as black, and historically have been considered
    # as such in American society.
    #
    # Increases black proportion from ~1.4% (all multiracial coded as multiracial) to ~2.3% (multiracial blacks coded as black)
    data["Ethnicity"].loc[data["Ethnicity"].str.contains("Black")] = "Black or of African descent"
    data["Ethnicity"].loc[data["Ethnicity"].str.split(";").map(lambda x: len(x) > 1) | (data["Ethnicity"] == "Biracial")] = "Multiracial"

    # Analogous coding for gender: anyone who picks multiple gender is coded as non-binary
    data["Gender"].loc[data["Gender"].str.split(";").map(lambda x: len(x) > 1)] = "Non-binary, genderqueer, or gender non-conforming"

    # Drop unneeded columns
    data = data[keep]

    # Some quick checks
    print(f'{len(data)} developers left in the sample after cleaning')
    print(f'{data.Ethnicity.str.contains("Black").sum()} or {data.Ethnicity.str.contains("Black").sum() / len(data)*100:.1f}% black developers in the sample with multiracial replacement')

    return data, keep, groups, categorical, numeric, base

def design_matrix(data, categorical, numeric, base):

    for cat in categorical:
        for col in list(set([i for row in data[cat].str.split(";") for i in row])):
            
            # Create control columns
            data[cat+"_"+col] = data[cat].str.split(";").map(lambda x: col in x)

        # Drop base level for each categorical
        data = data.drop(cat+"_"+base[cat], axis=1)

        # Drop original categorical column
        data = data.drop(cat, axis=1)

    # Creat cubic polynomials to model lifecycle dynamics across age and years of experience
    for col in numeric:
        if col != "Wage":
            data[col+"_2"] = data[col]**2
            data[col+"_3"] = data[col]**3

    data = data.astype("float")

    print(f'Design matrix complete with {data.shape[1]} variables/columns')
    
    X = data.drop(["Wage"], axis=1)
    Y = data["Wage"]

    return X, Y
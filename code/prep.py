import pandas as pd
import numpy as np

def text_clean(text: str):
    return text.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace(',', '_').replace('’', '').replace('.', '').replace(':', '').replace('___', '_').replace('__', '_')

def prep(data: pd.DataFrame, outcome: str, year: int):
    """
    Prepare DataFrame

    data: Pandas DataFrame with the survey data

    outcome: String name of outcome variable
    """
    
    data = data.copy()

    data = data.rename(columns={"WebFrameWorkedWith": "WebframeWorkedWith"})

    data["Year"] = str(year)

    # Columns to keep
    keep = [
        "Hobbyist",
        "Age",
        outcome,
        "DatabaseWorkedWith",
        "DevType",
        "EdLevel",
        "Employment",
        "Ethnicity",
        "Gender",
        "LanguageWorkedWith",
        "MiscTechWorkedWith",
        #"NEWCollabToolsWorkedWith",
        #"NEWDevOps",
        #"NEWLearn",
        #"NEWOvertime",
        "OpSys",
        "OrgSize",
        "PlatformWorkedWith",
        "PurchaseWhat",
        "Sexuality",
        "Trans",
        "UndergradMajor",
        "WebframeWorkedWith",
        "YearsCode",
        "YearsCodePro",
        "Year"
    ]

    # May be useful later
    groups = {
        "Demographics": ["Age", "Ethnicity", "Gender", "Sexuality", "Trans"],
        "Education": ["EdLevel", "UndergradMajor"],
        "Experience": ["DatabaseWorkedWith", "LanguageWorkedWith", "MiscTechWorkedWith", "NEWCollabToolsWorkedWith", "OpSys", 
                    "PlatformWorkedWith", "PurchaseWhat", "WebframeWorkedWith", "YearsCode", "YearsCodePro"],
        "Job": ["DevType", "Employment", "NEWOvertime"],
        "Personal_Professional": ["Hobbyist", "NEWLearn"], 
        "Company": ["NEWDevOps", "OrgSize"]
    }

    # Separate numeric and categorical columns
    numeric = ["Age", outcome, "YearsCode", "YearsCodePro"]
    categorical = keep.copy()

    for item in keep:
        if item in numeric:
            categorical.remove(item)

    data[categorical] = data[categorical].applymap(lambda x: str(x).replace("’", ""))

    # Base/omitted level for dummy variables
    base = {
        "Hobbyist": "No",
        "DatabaseWorkedWith": "MySQL",
        "DevType": "Developer, full-stack",
        "EdLevel": "Bachelors",
        "Employment": "Employed full-time",
        "Ethnicity": "Other",
        "Gender": "Man",
        "LanguageWorkedWith": "JavaScript",
        "MiscTechWorkedWith": "Node.js",
        #"NEWCollabToolsWorkedWith": "Jira",
        #"NEWDevOps": "No",
        #"NEWLearn": "Once every few years",
        #"NEWOvertime": "Never",
        "OpSys": "Windows",
        "OrgSize": "2 to 9 employees",
        "PlatformWorkedWith": "Windows",
        "PurchaseWhat": "I have little or no influence",
        "Sexuality": "Straight / Heterosexual",
        "Trans": "No",
        "UndergradMajor": "Computer science, computer engineering, or software engineering",
        "WebframeWorkedWith": "React.js",
        "Year": "2019"
    }

    for k in base.keys():
        base[k] = text_clean(base[k])

    # For this analysis, we are focused on:
    #   U.S. developers
    #   Professional developers
    #   At least 18 years of age
    #   Work at least 20 hours per week
    data = data[data["Country"] == "United States"]
    data = data[data["MainBranch"] == "I am a developer by profession"]
    data = data[data["Age"] >= 18]
    data = data[data["WorkWeekHrs"] >= 20]

    # Drop observations without years of coding experience given this is an important control
    data = data.dropna(subset=["YearsCode", "YearsCodePro"])

    for col in ["YearsCode", "YearsCodePro"]:
        data[col].loc[data[col] == "Less than 1 year"] = 0.5
        data[col].loc[data[col] == "More than 50 years"] = 51
        data[col] = data[col].astype("float")

    # Collapse responses with no college education to "No_college"
    data["EdLevel"].loc[(data["EdLevel"] == "I never completed any formal education") | \
                        (data["EdLevel"] == "Primary/elementary school") | \
                        (data["EdLevel"] == "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)")] = "No_college"

    # Clean up of various education levels
    for degree in ["Associate", "Bachelors", "Masters", "Other doctoral", "Professional", "Some college"]:
        data["EdLevel"].loc[data["EdLevel"].str.contains(degree)] = degree

    # Clean up of various majors
    for degree in ["Another engineering", "social science", "natural science", "health science", "humanities", "Fine arts", "business discipline"]:
        data["UndergradMajor"].loc[data["UndergradMajor"].str.contains(degree)] = degree

    # Cleanup `OrgSize` to be consistent across datasets
    data["OrgSize"].loc[data["OrgSize"] == "2-9 employees"] = "2 to 9 employees"
    data["OrgSize"].loc[data["OrgSize"].str.contains("Just me")] = "1 employee"

    # Clean up various tools
    for framework in ["Angular", "ASPNET"]:
        data["WebframeWorkedWith"].loc[data["WebframeWorkedWith"].str.contains(framework)] = framework

    for platform in ["Slack"]:
        data["PlatformWorkedWith"].loc[data["PlatformWorkedWith"].str.contains(platform)] = platform

    # Original dataset does poor job of imputing annualized earnings for those who entered "Monthly" or "Weekly" for CompFreq, as many of these respondents in fact put in 
    # their annual pay. This causes the ConvertedComp numbers to be badly inflated for these respondents, as it's taking annual values and multiplying them by 50 or 12.
    #
    # Since it's highly unlikely that someone who earns more than $200,000 per year would report their earnings on a weekly or monthly basis, we replace all those
    # responses with the value from CompTotal, which is the raw input from the survey. 
    data["ConvertedComp"].loc[(data["Country"] == "United States") & ((data["CompFreq"] == "Monthly") | (data["CompFreq"] == "Weekly")) & (data["ConvertedComp"] > 200000)] = \
        data["CompTotal"].loc[(data["Country"] == "United States") & ((data["CompFreq"] == "Monthly") | (data["CompFreq"] == "Weekly")) & (data["ConvertedComp"] > 200000)]
    
    # As we are dealing with self-reported income data, it is possible that some folks have lied about their income
    # with absurdly high or low values. In order to filter for this I remove some high and low values outside the 
    # typical earnings range for software developers. This is admittedly a coarse approach, but likely not far from
    # the optimal approach.
    data = data[(data["ConvertedComp"] <= 500000) & (data["ConvertedComp"] > 5000)]

    # We then transform the annual income values into houry earnings by dividing by 50 (which Stack Overflow assumes
    # as the number of working weeks) and then dividing by the self-reported number of work week hours
    if outcome == "Wage":
        data["Wage"] = np.log(data["ConvertedComp"] / 50 / data["WorkWeekHrs"])

    # Inflate wages to February 2020 values per https://www.bls.gov/opub/ted/2020/consumer-prices-increase-2-point-3-percent-for-year-ending-february-2020.htm
    data["Wage"] += (year == 2019) * np.log(1.023)

    # Given income is a major focus of the analysis, we drop the small number of respondents with missing income
    #
    # We then replace any other missing values with "No_answer", which we will explicitly control for later
    print(f'Removing {data[outcome].isna().sum()} respondents with missing {outcome}')
    data = data.dropna(subset=[outcome])
    
    data = data.fillna("No_answer")
    data = data.replace("nan", "No_answer")

    # Given the focus on black developers and the very small sample proportion, coding all developers who identify as at least
    # partially black as black. All other multiracial individuals coded as multiracial
    #
    # Anecdotally, many developers who are multiracial blacks self-indentify as black, and historically have been considered
    # as such in American society.
    #
    # Increases black proportion from ~1.4% (all multiracial coded as multiracial) to ~2.3% (multiracial blacks coded as black)
    data["Ethnicity"].loc[data["Ethnicity"].str.contains("Black")] = "Black or of African descent"
    data["Ethnicity"].loc[data["Ethnicity"].str.split(";").map(lambda x: len(x) > 1) | (data["Ethnicity"] == "Biracial")] = "Multiracial"
    data["Ethnicity"].loc[~data["Ethnicity"].str.contains("Black")] = "Other"

    # Analogous coding for gender: anyone who picks multiple gender is coded as non-binary
    data["Gender"].loc[data["Gender"].str.split(";").map(lambda x: len(x) > 1)] = "Non-binary, genderqueer, or gender non-conforming"

    # Drop unneeded columns
    data = data[keep]

    # Some quick checks
    print(f'{len(data)} developers left in the sample after cleaning')
    print(f'{data.Ethnicity.str.contains("Black").sum()} or {data.Ethnicity.str.contains("Black").sum() / len(data)*100:.1f}% black developers in the sample with multiracial replacement')

    return data, keep, groups, categorical, numeric, base

def design_matrix(data: pd.DataFrame, categorical: list, numeric: list, base: dict, outcome: str):
    """
    Create design matrix for regressions

    data: Cleaned pandas DataFrame

    categorical / numeric: Lists of separate variable types

    base: Base/omitted level for dummy variables

    outcome: String name of outcome variable
    """

    for cat in categorical:
        for col in sorted(set([i for row in data[cat].str.split(";") for i in row])):
            
            # Create control columns
            data[cat+"_"+text_clean(col)] = data[cat].str.split(";").map(lambda x: col in x)
            #data = data.rename(columns={cat+"_"+col: text_clean(cat+"_"+col)})

        # Drop base level for each categorical
        data = data.drop(cat+"_"+base[cat], axis=1)

        # Drop original categorical column
        data = data.drop(cat, axis=1)

    # Create cubic polynomials to model lifecycle dynamics across age and years of experience
    for col in numeric:
        if col != outcome:
            data[col+"_2"] = data[col]**2
            data[col+"_3"] = data[col]**3

    data = data.astype("float")

    print(f'Design matrix complete with {data.shape[1]} variables/columns')
    
    X = data.drop([outcome], axis=1)
    Y = data[outcome]

    return X, Y
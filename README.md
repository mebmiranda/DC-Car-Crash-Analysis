# D.C. Car Crash Analysis
Analysis of transportation-related crashes (car, motorcycle, pedestrian, bike) in the Washington, D.C. area.
This project is in-line with D.C.'s Vision Zero goal. The project is for partial fulfillment of the requirements for [CSE 6242 Data and Visual Analtytics](https://omscs.gatech.edu/cse-6242-data-visual-analytics).
<br>
## Problem Definition
Every year, car accidents are always among the top causes of death in Washington, D.C. In response, the city has launched its [Vision Zero Initiative](https://ddot.dc.gov/page/vision-zero-initiative), an effort to reduce vehicle-related crashes to zero by 2024. The team will determine the root causes of car accidents in Washington, D.C. by analyzing patterns, seasonality, and trends in the data collected from the [Vision Zero website](https://www.dcvisionzero.com/maps-data) and other sources. Our goal is to create an interactive and robust visualization dashboard to communicate the true impact of traffic-related crashes in Washington, D.C. to stakeholders who would be most interested in D.C.’s Vision Zero plan, such as policymakers, police, and local residents.


## Data
The table below lists all the data sources with their descriptions and a clickable link to where we obtained it:

| Dataset       | Description |
| ------------- | ------------- |
|[Vision Zero Data](https://www.dcvisionzero.com/maps-data)| The official website Vision Zero. The site contains the data related to Vision Zero.  |
| [Vision Zero Safety](https://opendata.dc.gov/datasets/DCGIS::vision-zero-safety/explore?location=38.911736%2C-77.034535%2C12.25)  | This dataset supports the Vision Zero Initiative and comes from a web-based application developed to allow the public to communicate the real and perceived dangers along the roadway from the perspective of either a pedestrian, bicyclist or motorists. (~5600 rows, 1.6 MB)|
|[Crashes in D.C.](https://opendata.dc.gov/datasets/crashes-in-dc/explore?location=38.893689%2C-77.019147%2C12.00&showTable=true)|This dataset represent the crash locations associated along the District of Columbia roadway blocks network. A companion crash details related table also exists for download. (~270,000 rows, 110 MB)|
|[Crash Table Data](https://opendata.dc.gov/datasets/DCGIS::crash-details-table/explore)|This table is a companion to the Crashes in DC layer. It is a related table containing details for each crash such as methods of transportation, some demographics for persons and injury types. (~720,000 rows, 66 MB)|
|[Automated Traffic Enforcement](https://opendata.dc.gov/datasets/automated-traffic-enforcement/explore?location=38.894716%2C-76.562079%2C10.57)|The Automated Traffic Enforcement (ATE) is a division of the District Department of Transportation (DDOT) that uses photo enforcement cameras as one of traffic calming measures to enforce traffic laws, and to reduce violations at DC’s streets and most intersections. (138 rows, 16 KB)| 

<br>


## Approaches
### Vision Zero DC Analytics Website
The team has cataloged its results in the following website, [Vision Zero DC Analytics](https://visionzerodcanalytics.godaddysites.com/). The site includes:
- **Home Page** | The group mission, which is to support D.C. Vision Zero initiative by creating interactive visualizations and machine learning models.
- **Dashboard** | All 4 of the team’s Tableau dashboards.
- **Machine Learning** | Machine learning blogs and full write-ups.
- **About Us** | The team member bios and team contact information.

### Dashboard
The team’s main Tableau workbook includes 4 dashboard views:
- **Demographics Dashboard** | Breaks down crashes by rider impairment, driver speeding, ticket
issuance, individual’s age, and license plate state (MD, VA, DC, and others).
- **Crash Analysis Dashboard** | Visualizes crashes by ward, person, and type of injury sustained. The
dashboard can be filtered by date, ward, person, injury type, and is fully interactive. Due to data
limitations, this dashboard contains data from 2015 onwards only.
- **Risk Analysis Dashboard** | Uses Bayesian Statistics to determine the probability of car crashes
given time, the likelihood of car crashes given day, and the possibility of car crashes per time given day. The dashboard can be filtered by year, quarter, month, ward, and street name. Due to data limitations, the dashboard only contains data from 2021 onwards.
- **Time Series Analysis Dashboard** | Shows the actual and forecasted number of fatal, major, minor, and unknown injuries over time. Moreover, the dashboard can be filtered by month, year, and on who was injured - driver, pedestrian, bicyclist, and passenger. It can be filtered by who is injured, street name, and ward. Due to data limitations, the dashboard only contains data from 2015 onwards.

### K-Means Clustering
The first approach used is a classical K-Means clustering algorithm. In this approach, the number of clusters is pre-set the number of clusters to see how cleanly the car crashes separate into respective clusters. The rationale is that DC has 8 pre-defined wards, which are similar to congressional districts in that the boundaries are assigned during each 10-year redistricting cycle such that each ward has roughly the same population. Setting a ‘K’ value of 8 illuminates the extent to which crash clusters correspond to existing ward boundaries. Since there can be multiple crashes at the same location, weighting locations by the number of crashes occurring in a given location provides a more meaningful understanding of crash concentration. Cluster separability is measured using an elbow plot of the sum of both weighted and unweighted squared differences between clusters.

### DBSCAN
The second approach used is DBSCAN, a density-based clustering approach. In this approach, the maximum distance that points can be set apart in order to be clustered together is pre-set. The benefit is that the number of clusters does not need to be pre-set. This allows the number of clusters to be assigned from model parameters subject to the structure of the data itself.

<br>

## Final Output/Conclusion
This study evaluated the D.C. car crash dataset, and used
interactive dashboards, machine learning models, and an easy-to-use website to help the D.C. government achieve its [Vision Zero DC Analytics](https://visionzerodcanalytics.godaddysites.com/). Users are able to utilize the resources the team has created directly from the website to analyze common demographics and risk factors associated with increased risks of car crashes. From our analysis, we determined that some of these primary risk factors include, but are not limited to: those with ticket violation(s) are 13.45% more likely to get involved in a car crash; Saturday evening is the most dangerous time to drive; people aged 25-30 are more predisposed to being involved in a crash; drivers from Maryland contributed to more crashes than drivers from any other state (including Washington, D.C.); and crashes are most prevalent in Wards 2, 5, and 7. Lastly, despite the team’s best efforts to use unsupervised clustering algorithms to look for geographic trends in the data, the data is unfortunately too dense to get meaningful results.

[Website Link](https://visionzerodcanalytics.godaddysites.com/)
<br>

<img width="800" alt="image" src="https://user-images.githubusercontent.com/82742020/207738850-2f3da380-d0d2-42b2-920d-cdba06350faa.png">



[Dashboard Link](https://public.tableau.com/app/profile/maynard.emmanuel.miranda/viz/2020USPRESIDENTIALELECTION_16561151719830/FinalDashboard)
<br>
<img width="803" alt="image" src="https://user-images.githubusercontent.com/82742020/207738670-aad9f6a1-aa70-4530-8c9c-f0ba6f6c8e4e.png">

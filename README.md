# D.C. Car Crash Analysis
Analysis of transportation-related crashes (car, motorcycle, pedestrian, bike) in the Washington, D.C. area.
This project is in-line with D.C.'s Vision Zero goal. The project is for partial fulfillment of the requirements for [CSE 6242 Data and Visual Analtytics](https://omscs.gatech.edu/cse-6242-data-visual-analytics).
<br>
## Problem Definition
Every year, car accidents are always among the top causes of death in Washington, D.C. In response, the city has launched its [Vision Zero Initiative](https://ddot.dc.gov/page/vision-zero-initiative), an effort to reduce vehicle-related crashes to zero by 2024. The team will determine the root causes of car accidents in Washington, D.C. by analyzing patterns, seasonality, and trends in the data collected from the [Vision Zero website](https://www.dcvisionzero.com/maps-data) and other sources. Our goal is to create an interactive and robust visualization dashboard to communicate the true impact of traffic-related crashes in Washington, D.C. to stakeholders who would be most interested in D.C.’s Vision Zero plan, such as policymakers, police, and local residents.
<br>

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
## Expected Innovation
Although D.C.’s Vision Zero website includes numerous data visualizations and analytical products, they
lack the following features, that we aim to improve upon:
- **Disjointed Visualizations** - Even though all of the visualizations are built in Tableau and portray
insightful information, they are not connected in one seamless dashboard, allowing users to see
relationships across multiple features at once.
- **Disparate Tools Used** - Currently, the Vision Zero Team creates its visualization products in both
Tableau and ArcGIS. This allows for more powerful mapping capabilities in ArcGIS; however, the
two products are kept separately, making it difficult to understand the relevance of the
geospatial data in the context of other visualizations.
- **Lack of Diversity in Datasets Used** -  Currently, only data on direct crashes are being used.
- **Insufficient Incorporation of Geographic Analysis** - While the current data is mapped, there is
substantial room for additional spatial analysis of existing crash data.
<br>
To build upon this, we will not only integrate multiple data sources, but we will centralize all the data and corresponding visualizations into Tableau. Because we will only be using Tableau, and it offers free licenses to those with University emails, the cost of this project is $0 (Q7). Additionally, we will use time-series modeling to project traffic-related crashes into the future. Lastly, the team plans to integrate not only crash/fatality data, but also demographic, socio-economic, and land use data such as the location of new road infrastructure.
<br>

## Intended Impact
We will integrate multiple data sources to see how different road calming measures (raised surfaces, slowed speed limits, etc.) affect crash frequency before/after implementation. If this effort, or other efforts like it, are unable to find effective methods of crash mitigation, more lives will continue to be at stake. Potential risks exist in the methods for pulling the data and also in joining data. For example, joining crash data, which occurs at a specific location, to road calming efforts, which usually occur in a broader area. 

<br>

For this effort to be successful, we will adhere to the aforementioned Plan of Activities and drive towards effective data integration of our various data sources, modeling of the data that provides meaningful insight into future trends, and development of our interactive dashboard. Success of this project will contribute to the success of D.C.’s goal of zero crash-related fatalities by 2024 in which the D.C. government will measure by the end of 2014.

## &nbsp;**DAX Measures – Hospital Analytics Project**



&nbsp;  Admissions Measures



&nbsp;  Total Admissions :-



&nbsp;  DAX



&nbsp; Total Admissions = COUNTROWS(FactVisit)



&nbsp;  

&nbsp;  Current Month Admissions =

&nbsp;  CALCULATE(

&nbsp;   \[Total Admissions],

&nbsp;   DATESMTD(DimDates\[Date])

&nbsp;   )



Last Month Admissions =

CALCULATE(

&nbsp;   \[Total Admissions],

&nbsp;   DATEADD(DimDates\[Date], -1, MONTH)

)



&nbsp; Revenue Measures



Total Revenue = SUM(FactVisit\[Revenue])



Revenue Growth % (MoM)2 = 

VAR CurrentRevenue = \[Total\_Revenue]

VAR PreviousRevenue =

&nbsp;   CALCULATE(

&nbsp;       \[Total\_Revenue],

&nbsp;       DATEADD(DimDates\[Date], -1, MONTH)

&nbsp;   )

RETURN

DIVIDE(CurrentRevenue - PreviousRevenue, PreviousRevenue)



Revenue Growth % (YoY) = 

VAR CurrentRevenue = \[Total\_Revenue]

VAR PreviousRevenue =

&nbsp;   CALCULATE(

&nbsp;       \[Total\_Revenue],

&nbsp;       SAMEPERIODLASTYEAR(DimDates\[Date])

&nbsp;   )

RETURN

DIVIDE(CurrentRevenue - PreviousRevenue, PreviousRevenue)



Avg Length of Stay = 

AVERAGE(FactVisit\[LengthOfStayDays])



Avg Revenue per Patient = 

DIVIDE(

&nbsp;   SUM(FactVisit\[TotalRevenue]),

&nbsp;   DISTINCTCOUNT(FactVisit\[PatientID])



Insurance Coverage % = 

DIVIDE(

&nbsp;   SUM(FactVisit\[InsuranceCovered]),

&nbsp;   SUM(FactVisit\[TotalRevenue]),

&nbsp;   0

)



Recovery Rate % = 

DIVIDE(

&nbsp;   CALCULATE(COUNTROWS(FactVisit), FactVisit\[Recovered] = "Yes"),

&nbsp;   COUNTROWS(FactVisit),

&nbsp;   0

)



Revenue MoM KPI Display = 

VAR Change = \[Revenue Growth % (MoM)2]

VAR Arrow =

&nbsp;   IF(Change >= 0, "▲ ", "▼ ")

VAR PercentText =

&nbsp;   FORMAT(Change, "0.0%")

RETURN

Arrow \& PercentText



Revenue YoY KPI Display = 

VAR Change = \[Revenue YoY %]

VAR Arrow =

&nbsp;   IF(Change >= 0, "▲ ", "▼ ")

VAR PercentText =

&nbsp;   FORMAT(Change, "0.0%")

RETURN

Arrow \& PercentText





Revenue YoY Color = 

IF(\[Revenue YoY %] >= 0, "Green", "Red")






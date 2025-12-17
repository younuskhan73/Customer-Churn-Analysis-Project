### **Tableau Calculated Fields and Measures**



**1.Headcount \& Employee Status**

 **1.Employee Count**

   **COUNT(\[Employee Number])**



 **2.Active Employees**

   **IF \[Attrition] = "No" THEN 1 ELSE 0 END**



 **3.Employees Left**

   **IF \[Attrition] = "Yes" THEN 1 ELSE 0 END**



 **4.Total Employees Left**

   **SUM(\[Employees Left])**



**2. Attrition \& Retention KPIs**

**Attrition Rate (%)**

**SUM(\[Employees Left]) / COUNT(\[Employee Number])**



**Retention Rate (%)**

**1 - ( SUM(\[Employees Left]) / COUNT(\[Employee Number]) )**



**Monthly Attrition Rate**

**SUM(\[Employees Left]) / COUNTD(\[Employee Number])**



**🔹 3. Compensation \& Salary Metrics**

**Average Monthly Income**

**AVG(\[Monthly Income])**



**Daily Rate**

**\[Monthly Income] / 30**



**Salary Band (Calculated Dimension)**

**IF \[Monthly Income] < 3000 THEN "Low"**

**ELSEIF \[Monthly Income] < 7000 THEN "Medium"**

**ELSE "High"**

**END**



**🔹 4. Tenure \& Experience Metrics**

**Employee Tenure (Years)**

**DATEDIFF('year',\[Hire Date],TODAY())**





**Average Tenure**

**AVG(\[Tenure])**



**🔹 5. Performance Metrics**

**High Performer Flag**

**IF \[Performance Rating] >= 4 THEN 1 ELSE 0 END**



**High Performer Ratio (%)**

**SUM(\[High Performer Flag]) / COUNT(\[Employee Number])**



**🔹 6. Workload \& Burnout Indicators**

**Overtime Flag**

**IF \[Over Time] = "Yes" THEN 1 ELSE 0 END**



**Overtime Percentage**

**SUM(\[Overtime Flag]) / COUNT(\[Employee Number])**



**Burnout Risk Indicator**

**IF \[Over Time] = "Yes" AND \[Job Satisfaction] <= 2 THEN "High Risk"**

**ELSE "Low Risk"**

**END**



**🔹 7. Engagement \& Satisfaction**

**Average Job Satisfaction**

**AVG(\[Job Satisfaction])**



**Low Satisfaction Flag**

**IF \[Job Satisfaction] <= 2 THEN 1 ELSE 0 END**



**🔹 8. Hiring \& Demographics**

**Average Age**

**AVG(\[Age])**



**Gender Diversity %**

**SUM( IF \[Gender] = "Female" THEN 1 ELSE 0 END ) / COUNT(\[Employee Number])**



**🔹 9.Department-wise Attrition Rate**

**SUM(\[Employees Left]) / COUNTD(\[Employee Number])**



**🔹 10. Box Plot Supporting Fields**

**Monthly Income (for Box Plot)**

**\[Monthly Income]**



**Tenure (for Box Plot)**



**\[Tenure]**





**I implemented custom calculated fields and LOD expressions in Tableau to derive attrition, compensation, and performance KPIs while maintaining employee-level granularity.**




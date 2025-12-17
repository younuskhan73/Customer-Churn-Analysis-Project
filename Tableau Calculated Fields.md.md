### **Tableau Calculated Fields**



**1. Base Metrics**

  **Total Revenue**

  **SUM(\[Revenue])**



  **Total Cost**

  **SUM(\[Cost])**



  **Total Profit**

  **SUM(\[Profit])**





 **2. Profitability KPIs**



   **Profit Margin (%)**

   **SUM(\[Profit]) / SUM(\[Revenue])**



   **Average Revenue**

   **AVG(\[Revenue])**



   **Average Profit**

   **AVG(\[Profit])**



**3. Time-Based Calculations**

  **Year**

  **YEAR(\[Order Date])**



  **Month**

  **DATETRUNC('month', \[Order Date])**



  **Year-to-Date (YTD) Revenue**

  **RUNNING\_SUM(SUM(\[Revenue]))**



  **Year-to-Date (YTD) Profit**

  **RUNNING\_SUM(SUM(\[Profit]))**



**4. Growth Metrics** 

   **Previous Year Revenue**

   **LOOKUP(SUM(\[Revenue]), -1)**



   **YoY Revenue Growth (%)**

   **(SUM(\[Revenue]) - LOOKUP(SUM(\[Revenue]), -1))**

   **/ LOOKUP(SUM(\[Revenue]), -1)**



  **YoY Profit Growth (%)**

 **(SUM(\[Profit]) - LOOKUP(SUM(\[Profit]), -1))**

  **/ LOOKUP(SUM(\[Profit]), -1)**



**5. Contribution \& Share Metrics**

  **Revenue Contribution (%)**

  **SUM(\[Revenue]) / TOTAL(SUM(\[Revenue]))**



  **Profit Contribution (%)**

  **SUM(\[Profit]) / TOTAL(SUM(\[Profit]))**



**6. Performance Indicators**

   **High Revenue Flag**

   **IF \[Revenue] > WINDOW\_AVG(SUM(\[Revenue])) THEN 1 ELSE 0 END**



  **Loss Indicator**

   **IF \[Profit] < 0 THEN "Loss" ELSE "Profit" END**



**7. LOD Expressions** 



   **Industry-Level Total Revenue**

  **{ FIXED \[Industry] : SUM(\[Revenue]) }**

 

  **Industry-Level Profit Margin**

  **{ FIXED \[Industry] :**

    **SUM(\[Profit]) / SUM(\[Revenue])**

   **}**



   **Customer-Level Revenue**

   **{ FIXED \[Customer ID] : SUM(\[Revenue]) }**



 **8.KPI Status Indicators** 

   **Revenue Performance Status**

   **IF SUM(\[Revenue]) >= WINDOW\_AVG(SUM(\[Revenue])) THEN "Above Target"**

   **ELSE "Below Target"**

   **END**



  **Profit Health Indicator**

  **IF SUM(\[Profit]) > 0 THEN "Healthy" ELSE "At Risk" END**










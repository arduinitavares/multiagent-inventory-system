# Project Reflection: Multi-Agent System for Beaver's Choice Paper Company

This document provides a comprehensive reflection on the design, performance, and future improvements for the multi-agent inventory system.

---

## Architectural Decisions and Orchestration
The system is built on an **orchestrator-worker architecture**, chosen for its **modularity** and **clear separation of concerns**.  

- A central **Customer Service Orchestrator** manages the overall workflow.  
- It delegates tasks to three specialized worker agents:  
  - **Inventory Manager**  
  - **Sales Quoter**  
  - **Order Processor**  

The orchestrator also maintains the **conversational state** (`INQUIRY`, `QUOTED`, `FULFILLED`) for each user session. This ensures that tasks are executed in a logical sequence—for example, **preventing a quote from being generated for an out-of-stock item**.  

Because each worker agent has a **single responsibility** and does not need awareness of the broader business process, the system is both **robust** and **predictable**.

---

## Evaluation Summary and System Strengths
The system was evaluated against a set of **10 sample requests**, with results logged in `test_results.csv`.  

A final step was added to explicitly invoke the **`generate_financial_report_tool`**, ensuring all required helper functions were included in the operational workflow.  

### Key Performance Metrics
- **Total Requests Processed:** 10/10 (100%)  
- **Successful Fulfillments:** 7/10 orders completed (including partial fulfillments).  
- **Total Cash Delta:**  
  - Initial balance: `$50,000.00`  
  - Final balance: `$46,209.70`  
  - Net change: **–$3,790.30**  
  - This is expected since inventory is depleted without restocking during the test run.  
- **Common Failure Reasons:** All unfulfilled requests were due to items being **out of stock**.  
  - Customers were notified of the next availability date.  
  - Orders continued with available items when possible.  

### System Strengths
- **Transparency:** Quotes include rationales, increasing customer trust.  
- **Resilience:** Handles partial or unfulfillable requests without failing the entire workflow.  

---

## Two Concrete Improvements for Future Implementation

### 1. Proactive Inventory Replenishment Agent
Introduce a **ProcurementAgent** to reduce stockouts:  
- Runs periodically (e.g., daily).  
- Uses `get_stock_level` across all items.  
- Compares current inventory with `min_stock_level`.  
- If below threshold → creates a **procurement transaction** to simulate supplier reordering.  

This ensures stock availability is proactively maintained.  

---

### 2. Persistent Backorder and Notification System
Enhance handling of out-of-stock requests:  
- When items are unavailable, the **Orchestrator** offers the user the option to **place a backorder**.  
- If accepted:  
  - **FulfillmentAgent** creates a **pending transaction**.  
  - **ProcurementAgent** monitors pending orders.  
  - Once restocked, pending orders are **automatically fulfilled**, and the customer is notified.  

This system would increase user satisfaction and reduce lost sales opportunities.  

---

## Closing Thoughts
The system has demonstrated strong **operational reliability** and **explainability**.  
With the addition of a ProcurementAgent and a Backorder System, it would evolve from a **reactive workflow manager** into a **proactive and customer-centric business platform**.

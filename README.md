# Multi-Agent System for Beaver's Choice Paper Company

## Project Overview
This project implements a multi-agent system to automate the inventory management, quoting, and order fulfillment processes for the **Beaver's Choice Paper Company**. The system is designed to handle customer inquiries via a text-based interface, providing a streamlined and efficient workflow from initial request to final sale confirmation.

The architecture is built using the **pydantic-ai** framework and consists of a central **orchestrator agent** that delegates tasks to three specialized **worker agents**. This modular design ensures a clear separation of concerns and robust handling of business logic.

---

## 1. Explanation of Agent Workflow and Architecture
The system's architecture follows a **classic orchestrator-worker model**, ideal for managing structured, multi-step business processes. This design ensures clarity, modularity, and scalability. Each agent has a single, well-defined responsibility, simplifying development, debugging, and future maintenance.

The workflow is visualized in `workflow_diagram.png` and detailed below.

### Agent Roles and Responsibilities
The system is composed of four distinct agents:

- **Customer Service Orchestrator**  
  - Serves as the "brain" of the operation.  
  - Interfaces directly with the user.  
  - Manages conversation states: `INQUIRY`, `QUOTED`, `FULFILLED`.  
  - Delegates tasks to worker agents.  
  - Synthesizes information into professional user responses.  

- **Inventory Manager**  
  - Provides accurate information about inventory.  
  - Uses tools: `get_stock_level`, `get_supplier_delivery_date`, `get_all_inventory`.  
  - Returns stock status and alternatives.  
  - Does **not** handle pricing or fulfillment.  

- **Sales Quoter**  
  - Generates price quotes once stock is confirmed.  
  - Uses `search_quote_history` to support pricing rationale.  
  - Applies bulk discount policy.  
  - Returns a structured **Quote object** with explanations.  

- **Order Processor (Fulfillment Agent)**  
  - Finalizes confirmed orders.  
  - Uses `create_transaction` to commit stock and record the sale.  
  - Returns official confirmation to the orchestrator.  

---

### Decision-Making and Data Flow
The system enforces a **sequential, state-driven flow** to prevent errors like selling unavailable items:

1. **User** sends a request.  
2. **Orchestrator** parses request, normalizes units (e.g., sheets → reams with `convert_sheets_to_reams`), and queries Inventory Manager.  
3. **Inventory Manager** checks stock database.  
4. If available → **Sales Quoter** generates structured quote.  
5. **Orchestrator** presents the quote and transitions to `QUOTED`.  
6. If user accepts → **Order Processor** finalizes transaction via `create_transaction`.  
7. **Orchestrator** confirms order and updates cash balance, transitioning to `FULFILLED`.  

---

## 2. Evaluation and Performance Analysis
The system was evaluated using **20 scenarios** from `quote_requests_sample.csv`. Results were logged in `test_results.csv`.  

### Key Findings
- **Successful Task Completion & State Management**  
  - All 20 requests were processed correctly.  
  - Quotes were generated, orders finalized, and cash balances updated.  

- **Robust Handling of Out-of-Stock Items**  
  - Example: Request #2 ("Party Streamers") flagged as out of stock.  
  - System suggested "Colored Paper" as alternative and completed a partial order.  

- **Transparency and Explainability**  
  - Example: Request #3 provided pricing rationale:  
    *"Aligned with past average of $1106.40 over 10 quotes"*.  
  - Improves trust and explainability for customers.  

- **Data Normalization & User-Friendliness**  
  - Example: Request #1 converted "200 sheets" → "1 ream".  
  - Ensures consistent quoting and fulfillment.  

---

## 3. Suggestions for Further Improvements

### 3.1 Implement a Proactive **Business Advisor Agent**
Currently, the system is **reactive**. A fifth agent could proactively analyze business data:  
- Identify frequently out-of-stock items → raise `min_stock_level`.  
- Detect slow-moving products → recommend discounts.  
- Spot customer request trends (e.g., eco-friendly paper) → suggest product line expansion.  

This would evolve the system into a **strategic business tool**.  

---

### 3.2 Develop an Advanced **Customer Interaction Agent**
Enhance interaction beyond "Yes, proceed":  

- **Negotiation:** Offer small one-time discounts (e.g., 5%) to close large deals.  
- **Clarification:** Handle vague requests (e.g., "paper for a big party") by asking follow-up questions.  
- **Personalization:** Maintain long-term memory of repeat customers for loyalty discounts or tailored offers.  

This would make the system more **flexible, human-like, and effective** at driving sales.  

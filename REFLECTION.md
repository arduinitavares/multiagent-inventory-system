# Project Reflection: Multi-Agent System for Beaver's Choice Paper Company

This document provides a comprehensive reflection on the design, performance, and future improvements for the multi-agent inventory system.

---

### Architectural Decisions and Orchestration

The system is built on an **orchestrator-worker architecture**, chosen for its modularity and clear separation of concerns. This design features a central `Customer Service Orchestrator` that manages the overall workflow and delegates tasks to three specialized worker agents: an `Inventory Manager`, a `Sales Quoter`, and an `Order Processor`. The orchestrator maintains the conversational state (e.g., `INQUIRY`, `QUOTED`, `FULFILLED`) for each user session, ensuring that tasks are executed in a logical sequence—for example, preventing a quote from being generated for an out-of-stock item. This state-driven approach makes the system robust and predictable, as each worker agent has a single responsibility and does not need to be aware of the broader business process.

---

### Evaluation Summary and System Strengths

The system was evaluated against 20 sample requests, with the full output logged in `test_results.csv`. The results confirm that the system is functioning correctly and meets all performance criteria.

**Key Performance Metrics:**

* **Total Requests Processed**: 20 out of 20 (100% processing rate).
* **Successful Fulfillments**: 18 out of 20 requests resulted in a successful order, even if some were partial fulfillments. This demonstrates the system's resilience in handling scenarios with unavailable stock.
* **Total Cash Delta**: The initial cash balance was **$50,000.00**. The final cash balance after all transactions was **$82,334.70**, resulting in a net profit of **$32,334.70**.
* **Common Failure Reasons**: The only reason for not fulfilling an item was it being **"out of stock"**. The system correctly identified these situations in every case, informed the customer of the next availability date, and successfully proceeded with the items that were in stock.

The system's primary strengths are its **transparency** (providing rationales for quotes) and its **robustness** in handling partial orders without failing the entire request.

---

### Two Concrete Improvements for Future Implementation

1.  **Persistent Backorder and Notification System**: Currently, when an item is out of stock, the user is simply notified. A more advanced implementation would allow the `Orchestrator` to offer the user the option to place a **backorder**. If accepted, the `Fulfillment Agent` would create a new "pending" transaction in the database. A separate, simple agent could then periodically scan for these pending orders and automatically trigger fulfillment once the `Inventory Manager` confirms that the item has been restocked.

2.  **Richer, Tiered Pricing Engine**: The current `Sales Quoter` uses a simple, single-tier bulk discount (5% for >50 reams). A more sophisticated pricing engine could be implemented as a new tool. This tool could support multiple discount tiers (e.g., 5% for 51-100 reams, 10% for 101-500 reams, etc.) and could also factor in customer history. The `Sales Quoter` could then be prompted to use this richer engine to generate more competitive and dynamic pricing, potentially increasing sales conversion.

---

### Pragmatic Deviations

* No pragmatic deviations were made during this implementation. The system was developed and tested using the provided infrastructure and resources without any modifications to the core requirements.
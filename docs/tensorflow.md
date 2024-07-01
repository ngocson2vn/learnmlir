# TensorFlow MLIR Dialects
https://github.com/tensorflow/community/blob/master/rfcs/20190612-mlir-dialect.md

## Islands of Predictability
"Islands of predictability" is a term often used in the context of complex systems, such as software engineering, machine learning, and cognitive science. It refers to the concept that within a complex, unpredictable system, there can be localized areas or components that exhibit predictable behavior.

In software engineering, for instance, this can mean that while the overall system might be highly complex and its behavior hard to predict, certain parts of the system (such as well-defined modules or components) may exhibit predictable patterns and behavior. Identifying and understanding these islands can help in managing and controlling the complexity of the entire system.

In machine learning, it can refer to parts of a model or data that exhibit consistent and predictable patterns, despite the overall complexity of the data or the model.

The idea is that by focusing on these predictable components or areas, one can make the larger system more manageable and better understand its behavior.

**A specific example in software engineering:**
Let's consider a large-scale web application as an example.

Imagine you have a web application with many components, such as user authentication, data processing, reporting, and user interfaces. The entire system is complex and has many interacting parts, making it challenging to predict how changes will affect the whole application.

However, within this complexity, there may be certain "islands of predictability" in the system. For example:

Example: Payment Processing Module  
Context:  
The application has a payment processing module that handles transactions. This module interacts with various external services and databases, but its core functionality is fairly isolated from the rest of the application.

Islands of Predictability:

- 1. Transaction Logic:
  The payment processing module has well-defined transaction logic. For instance, it consistently performs the same steps to process a payment: validate payment information, authorize the transaction, and update the user's account balance. These steps are predictable and can be tested independently.

- 2. API Contract:
  The module exposes a specific API for payment processing. This API has a stable contract (input parameters, expected output, error handling) that doesn't change frequently. This makes it easier to predict how the module will interact with other parts of the application or external systems.

- 3. Error Handling:
  The module has well-defined error handling mechanisms. It handles different types of errors (e.g., network issues, insufficient funds) in a predictable manner and returns consistent error messages.

Impact:

By understanding that the payment processing module has these islands of predictability, developers can confidently make changes or improvements to other parts of the system, knowing that the payment processing behavior will remain stable. Additionally, this predictability allows for easier testing and debugging of the payment-related features.

This isolated predictability helps manage the overall complexity of the web application, as changes in the payment processing module can be more easily contained and understood, without affecting the broader system unpredictably.
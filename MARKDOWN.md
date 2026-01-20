# ML Pipeline Development
###  ML Pipeline tracking using **DVC**
---
1. Data Ingestion 
- The process of collecting, importing, and transferring raw data from multiple sources into a centralized system (data lake, data warehouse, or feature store) where it can be processed for machine learning.
2. Data preprocessing
- The systematic process of cleaning, transforming, validating, and structuring raw data so it becomes suitable for machine learning algorithms.<br>

    a. **Data cleaning** <br>
    - Data cleaning removes errors, inconsistencies, and noise that distort learning. It includes handling missing values (deletion, mean/median/mode imputation), correcting incorrect entries, removing duplicates, and filtering out irrelevant records. Clean data ensures statistical assumptions are not violated during training.

    b. **Handling Missing Data**<br>
    - Missing data is addressed through imputation techniques (mean, median, regression, KNN) or row/column removal when absence is systematic. The strategy depends on data distribution and missingness type (MCAR, MAR, MNAR). Incorrect handling introduces bias and variance inflation.

    c. **Outlier Detection and Treatment**
    - Outliers are extreme values that can skew model parameters. Detection methods include Z-score, IQR, and model-based approaches. Treatment involves removal, capping, or transformation depending on domain relevance. Outlier handling stabilizes loss functions and gradient updates.

    d. **Data Transformation**

    e. **Feature Scaling**
    - Bringing all numerical features into similar ranges mostly in decimal palces for faster convergence. If one feature has values in thousands and another in decimals, many ML algorithms will incorrectly treat the larger-scale feature as more important. The methods are **z-score** where center data around zero with min max value within -1 and 1 or unit variance. The other method is **normalization (min-max)** where the values are rescaleds to fixed range between 0 and 1. Distance-based models (KNN, SVM), gradient-based models (linear/logistic regression), and regularized models are sensitive to scale because they rely on distances or coefficient sizes during optimization. Without scaling, these models learn biased and unstable patterns.<br>

    f.  **Endcoding Categorical Varaibles**
    - Categorical data is converted into numeric form using label encoding, one-hot encoding, ordinal encoding, or target encoding. The encoding choice affects dimensionality, sparsity, and information leakage. Proper encoding preserves semantic meaning.

    g. **Feature Engineering**
    - Feature engineering creates new informative variables using domain knowledge (aggregations, interactions, ratios, time-based features). This step often contributes more to performance gains than model selection itself.
    
    h. **Feature Selection**
    -  Feature selection means keeping only the useful input variables and removing the ones that do not help prediction. Some features add no new information or repeat what other features already explain, which can confuse the model instead of helping it. <br>
    By removing these unnecessary features, the model becomes simpler, faster, and easier to understand. Fewer features reduce noise, help the model focus on real patterns, and prevent it from memorizing the data (overfitting) instead of learning general rules
    
    i. **Data Splitting**
    - Data is divided into training, validation, and test sets to prevent information leakage and enable unbiased evaluation.
    
    j. **Data Validation**
    - Data validation means checking that incoming data still looks the way the model expects. It verifies that columns exist (schema), values stay within allowed limits (range checks), data types are correct, and data patterns have not changed over time (distribution drift). These checks catch problems early before they silently damage model predictions.

    k. **EDA**
    - EDA is the decision-making layer of an ML pipeline. It does not modify data; it explains it. Every preprocessing step—scaling, transformation, feature selection—should be justified by insights from EDA. Skipping EDA leads to fragile models built on unchecked assumptions. Strong ML systems treat EDA as mandatory, not optional.
3. Model Building
4. Model Evaluation with **mlflow** - Save all the artificats in mlflow to track later on
5. Model Registration in mlflow and create API with help of flask  and use the model, develop chrom plugin.

**NOTE: Create repository and clone the repository and clone the repository in particular local directory.**

### To see the files and folder icons installed extension called **Material Icon Theme** from Extensions. 
![alt text](image.png)

Step 1: Create virtual environment and activate
```python
python -m venv .venv
.venv\Scripts\activate
```
Step 2: Install requirement.txt
```python
pip install -r requirements.txt

pip install -r requirements.txt --timeout 1000


# Command to see all the installed libraries/modules
pip list
```
Step 3: git init

Step 4: dvc init
- Initialization sets up basic rules and structures so system knows how to work before doing any real task. dvc init means initializing Data Version Control (DVC) in your project so the project knows how to track data, models, and ML pipelines in a structured way. It is a one-time setup step that tells the project environment:

Step to check dvc 


# **FLASK**
```python
app = Flask(__name__)
```
- Flask is just a blueprint for creating web applications,suggest how to buld application, how to send and receive responses, but Flask does nothing until instance of the class is created.

Hierarchical structure shows the Flask package organization with classes and their respective methods/functions organized in a tree-like format for easy understanding of the package architecture.
```Python
flask/
│
├── __init__.py
│   │
│   ├── Flask (class)
│   │   ├── __init__()
│   │   ├── route()
│   │   ├── add_url_rule()
│   │   ├── endpoint()
│   │   ├── before_request()
│   │   ├── before_first_request()
│   │   ├── after_request()
│   │   ├── teardown_request()
│   │   ├── teardown_appcontext()
│   │   ├── context_processor()
│   │   ├── url_value_preprocessor()
│   │   ├── url_defaults()
│   │   ├── errorhandler()
│   │   ├── register_error_handler()
│   │   ├── template_filter()
│   │   ├── template_test()
│   │   ├── template_global()
│   │   ├── make_response()
│   │   ├── run()
│   │   ├── test_client()
│   │   ├── test_request_context()
│   │   ├── app_context()
│   │   ├── request_context()
│   │   ├── register_blueprint()
│   │   ├── create_jinja_environment()
│   │   ├── create_url_adapter()
│   │   ├── dispatch_request()
│   │   ├── full_dispatch_request()
│   │   ├── handle_exception()
│   │   ├── handle_http_exception()
│   │   ├── handle_user_exception()
│   │   ├── log_exception()
│   │   ├── open_session()
│   │   ├── save_session()
│   │   ├── make_config()
│   │   ├── make_default_options_response()
│   │   ├── should_ignore_error()
│   │   ├── ensure_sync()
│   │   ├── async_to_sync()
│   │   ├── make_shell_context()
│   │   ├── shell_context_processor()
│   │   ├── Properties:
│   │   │   ├── config
│   │   │   ├── logger
│   │   │   ├── jinja_env
│   │   │   ├── got_first_request
│   │   │   ├── debug
│   │   │   ├── testing
│   │   │   ├── secret_key
│   │   │   ├── session_cookie_name
│   │   │   ├── permanent_session_lifetime
│   │   │   ├── blueprints
│   │   │   ├── extensions
│   │   │   ├── url_map
│   │   │   ├── subdomain_matching
│   │   │   ├── template_folder
│   │   │   ├── root_path
│   │   │   ├── name
│   │   │   └── import_name
│   │   │
│   │   ├── Request (class)
│   │   │   ├── Properties:
│   │   │   │   ├── args
│   │   │   │   ├── form
│   │   │   │   ├── files
│   │   │   │   ├── cookies
│   │   │   │   ├── headers
│   │   │   │   ├── method
│   │   │   │   ├── endpoint
│   │   │   │   ├── blueprint
│   │   │   │   ├── view_args
│   │   │   │   ├── url
│   │   │   │   ├── base_url
│   │   │   │   ├── url_root
│   │   │   │   ├── path
│   │   │   │   ├── script_root
│   │   │   │   ├── environ
│   │   │   │   ├── data
│   │   │   │   ├── json
│   │   │   │   ├── is_json
│   │   │   │   ├── is_secure
│   │   │   │   ├── scheme
│   │   │   │   └── mimetype
│   │   │   │
│   │   │   ├── get_json()
│   │   │   ├── get_data()
│   │   │   ├── on_json_loading_failed()
│   │   │   └── close()
│   │   │
│   │   ├── Response (class)
│   │   │   ├── __init__()
│   │   │   ├── set_cookie()
│   │   │   ├── delete_cookie()
│   │   │   ├── set_data()
│   │   │   ├── make_conditional()
│   │   │   ├── add_etag()
│   │   │   ├── Properties:
│   │   │   │   ├── status_code
│   │   │   │   ├── status
│   │   │   │   ├── headers
│   │   │   │   ├── data
│   │   │   │   ├── mimetype
│   │   │   │   ├── content_type
│   │   │   │   ├── content_length
│   │   │   │   ├── is_streamed
│   │   │   │   ├── is_sequence
│   │   │   │   └── direct_passthrough
│   │   │   │
│   │   │   ├── force_type()
│   │   │   └── from_app()
│   │   │
│   │   ├── Blueprint (class)
│   │   │   ├── __init__()
│   │   │   ├── route()
│   │   │   ├── add_url_rule()
│   │   │   ├── endpoint()
│   │   │   ├── before_request()
│   │   │   ├── before_app_request()
│   │   │   ├── before_app_first_request()
│   │   │   ├── after_request()
│   │   │   ├── after_app_request()
│   │   │   ├── teardown_request()
│   │   │   ├── teardown_app_request()
│   │   │   ├── context_processor()
│   │   │   ├── app_context_processor()
│   │   │   ├── errorhandler()
│   │   │   ├── register_error_handler()
│   │   │   ├── app_errorhandler()
│   │   │   ├── url_value_preprocessor()
│   │   │   ├── url_defaults()
│   │   │   ├── app_url_value_preprocessor()
│   │   │   ├── app_url_defaults()
│   │   │   ├── template_filter()
│   │   │   ├── app_template_filter()
│   │   │   ├── template_test()
│   │   │   ├── app_template_test()
│   │   │   ├── template_global()
│   │   │   ├── app_template_global()
│   │   │   ├── record()
│   │   │   ├── record_once()
│   │   │   ├── make_setup_state()
│   │   │   ├── register()
│   │   │   ├── register_blueprint()
│   │   │   └── Properties:
│   │   │       ├── name
│   │   │       ├── import_name
│   │   │       ├── static_folder
│   │   │       ├── static_url_path
│   │   │       ├── template_folder
│   │   │       ├── url_prefix
│   │   │       ├── subdomain
│   │   │       └── root_path
│   │   │
│   │   ├── Config (class)
│   │   │   ├── __init__()
│   │   │   ├── from_envvar()
│   │   │   ├── from_pyfile()
│   │   │   ├── from_object()
│   │   │   ├── from_file()
│   │   │   ├── from_mapping()
│   │   │   ├── from_prefixed_env()
│   │   │   ├── get_namespace()
│   │   │   └── Methods from dict (inherited)
│   │   │
│   │   ├── Functions (module-level):
│   │   │   ├── render_template()
│   │   │   ├── render_template_string()
│   │   │   ├── get_template_attribute()
│   │   │   ├── url_for()
│   │   │   ├── redirect()
│   │   │   ├── abort()
│   │   │   ├── make_response()
│   │   │   ├── after_this_request()
│   │   │   ├── send_file()
│   │   │   ├── send_from_directory()
│   │   │   ├── safe_join()
│   │   │   ├── escape()
│   │   │   ├── flash()
│   │   │   ├── get_flashed_messages()
│   │   │   └── jsonify()
│   │   │
│   │   └── Global Objects (LocalProxy):
│   │       ├── current_app
│   │       ├── g
│   │       ├── request
│   │       └── session
│   │
├── app.py
│   └── Flask (class) - main implementation
│
├── blueprints.py
│   ├── Blueprint (class)
│   └── BlueprintSetupState (class)
│       ├── __init__()
│       ├── add_url_rule()
│       └── Properties
│
├── config.py
│   └── Config (class)
│
├── ctx.py
│   ├── AppContext (class)
│   │   ├── __init__()
│   │   ├── push()
│   │   ├── pop()
│   │   └── __enter__() / __exit__()
│   │
│   ├── RequestContext (class)
│   │   ├── __init__()
│   │   ├── push()
│   │   ├── pop()
│   │   ├── auto_pop()
│   │   ├── match_request()
│   │   └── __enter__() / __exit__()
│   │
│   ├── Functions:
│   │   ├── has_request_context()
│   │   ├── has_app_context()
│   │   └── copy_current_request_context()
│   │
│   └── after_this_request() - function
│
├── helpers.py
│   ├── Functions:
│   │   ├── get_debug_flag()
│   │   ├── get_env()
│   │   ├── get_load_dotenv()
│   │   ├── stream_with_context()
│   │   ├── make_response()
│   │   ├── url_for()
│   │   ├── get_template_attribute()
│   │   ├── flash()
│   │   ├── get_flashed_messages()
│   │   ├── send_file()
│   │   ├── send_from_directory()
│   │   ├── safe_join()
│   │   ├── total_seconds()
│   │   └── is_ip()
│   │
│   └── locked_cached_property (class)
│       └── __get__()
│
├── json/
│   ├── __init__.py
│   │   ├── Functions:
│   │   │   ├── jsonify()
│   │   │   ├── dumps()
│   │   │   ├── dump()
│   │   │   ├── loads()
│   │   │   └── load()
│   │   │
│   │   └── JSONEncoder (class)
│   │       ├── default()
│   │       └── encode()
│   │
│   ├── provider.py
│   │   └── DefaultJSONProvider (class)
│   │       ├── dumps()
│   │       ├── loads()
│   │       ├── dump()
│   │       ├── load()
│   │       └── response()
│   │
│   └── tag.py
│       ├── TaggedJSONSerializer (class)
│       │   ├── tag()
│       │   ├── untag()
│       │   ├── dumps()
│       │   └── loads()
│       │
│       └── JSONTag (class - base)
│           ├── check()
│           ├── to_json()
│           └── to_python()
│
├── sessions.py
│   ├── SessionMixin (class)
│   │   └── Properties:
│   │       ├── permanent
│   │       ├── new
│   │       ├── modified
│   │       └── accessed
│   │
│   ├── SecureCookieSession (class)
│   │   └── Inherits from SessionMixin
│   │
│   ├── NullSession (class)
│   │   └── Inherits from SecureCookieSession
│   │
│   ├── SessionInterface (class - base)
│   │   ├── make_null_session()
│   │   ├── is_null_session()
│   │   ├── get_cookie_domain()
│   │   ├── get_cookie_path()
│   │   ├── get_cookie_httponly()
│   │   ├── get_cookie_secure()
│   │   ├── get_cookie_samesite()
│   │   ├── get_expiration_time()
│   │   ├── should_set_cookie()
│   │   ├── open_session()
│   │   ├── save_session()
│   │   └── pickle_based
│   │
│   └── SecureCookieSessionInterface (class)
│       ├── digest_method
│       ├── salt
│       ├── serializer
│       ├── session_class
│       ├── get_signing_serializer()
│       ├── open_session()
│       └── save_session()
│
├── signals.py
│   ├── Namespace (class)
│   │   └── signal()
│   │
│   └── Signals:
│       ├── template_rendered
│       ├── before_render_template
│       ├── request_started
│       ├── request_finished
│       ├── got_request_exception
│       ├── request_tearing_down
│       ├── appcontext_tearing_down
│       ├── appcontext_pushed
│       ├── appcontext_popped
│       └── message_flashed
│
├── templating.py
│   ├── Functions:
│   │   ├── render_template()
│   │   ├── render_template_string()
│   │   ├── get_template_attribute()
│   │   └── stream_template()
│   │
│   ├── Environment (class)
│   │   └── Extends Jinja2 Environment
│   │
│   └── DispatchingJinjaLoader (class)
│       └── Implements template loading
│
├── testing.py
│   ├── EnvironBuilder (class)
│   │   └── Werkzeug environment builder
│   │
│   ├── FlaskClient (class)
│   │   ├── __init__()
│   │   ├── open()
│   │   ├── session_transaction()
│   │   └── HTTP methods:
│   │       ├── get()
│   │       ├── post()
│   │       ├── put()
│   │       ├── delete()
│   │       ├── patch()
│   │       ├── options()
│   │       ├── head()
│   │       └── trace()
│   │
│   └── FlaskCliRunner (class)
│       ├── __init__()
│       └── invoke()
│
├── views.py
│   ├── View (class)
│   │   ├── decorators
│   │   ├── methods
│   │   ├── provide_automatic_options
│   │   ├── dispatch_request() - abstract
│   │   └── as_view() - classmethod
│   │
│   └── MethodView (class)
│       ├── Inherits from View
│       ├── dispatch_request()
│       └── HTTP method handlers:
│           ├── get()
│           ├── post()
│           ├── put()
│           ├── delete()
│           ├── patch()
│           ├── options()
│           └── head()
│
├── cli.py
│   ├── FlaskGroup (class)
│   │   ├── __init__()
│   │   ├── get_command()
│   │   ├── list_commands()
│   │   └── make_context()
│   │
│   ├── AppGroup (class)
│   │   └── Inherits from click.Group
│   │
│   ├── ScriptInfo (class)
│   │   ├── __init__()
│   │   ├── load_app()
│   │   └── create_app
│   │
│   ├── Functions:
│   │   ├── with_appcontext()
│   │   ├── pass_script_info()
│   │   ├── load_dotenv()
│   │   ├── show_server_banner()
│   │   ├── find_best_app()
│   │   ├── prepare_import()
│   │   └── locate_app()
│   │
│   └── Commands:
│       ├── run_command
│       ├── shell_command
│       └── routes_command
│
├── globals.py
│   ├── LocalProxy objects:
│   │   ├── current_app
│   │   ├── g
│   │   ├── request
│   │   └── session
│   │
│   └── LocalStack (class)
│       ├── push()
│       ├── pop()
│       └── top
│
├── logging.py
│   ├── Functions:
│   │   ├── has_level_handler()
│   │   ├── create_logger()
│   │   └── wsgi_errors_stream()
│   │
│   └── default_handler (StreamHandler)
│
├── wrappers.py
│   ├── Request (class)
│   │   └── Extends Werkzeug Request
│   │
│   └── Response (class)
│       └── Extends Werkzeug Response
│
└── debughelpers.py
    ├── FormDataRoutingRedirect (class)
    ├── attach_enctype_error_multidict()
    ├── Functions:
    │   └── explain_template_loading_attempts()
    │
    └── DebugFilesKeyError (class)
```

All the files in the above hierarchial order will be installed in the local machine when **!pip install flask** is performed.
- When you write **from flask import Flask**, Python goes to the installed flask package. The python goes to **__init__.py** file and look for the Flask class in the app.py, creatin short cut method to import otherwise import would look like **from flask.app import Flask**.
- Libraries such as pandas or NumPy operate in a direct, function-based manner where data is passed in, processed, and returned as output, without needing awareness of the surrounding project or filesystem. Flask, on the other hand, is a web framework that manages a long-running application which waits for external HTTP requests and serves responses dynamically, often involving templates, static files, configuration, and extensions. Because Flask must locate and manage these resources on disk, it needs to know where the application is defined. Passing __name__ to Flask(__name__) allows Flask to identify the module that created the application, determine the project’s root directory, and correctly resolve paths for templates, static files, instance configuration, and CLI discovery. Thus, __name__ does not handle data transfer itself, but instead tells Flask where the application lives, enabling the framework to manage requests and resources correctly at runtime.
- If you define app = Flask(__name__) inside main.py, then __name__ refers to the Python module name associated with main.py. Flask uses this module name to locate the underlying file (main.py) and from that determine the directory in which it is stored. That directory becomes the application’s root path, which Flask uses to resolve templates, static files, instance configuration, and other filesystem-based resources. In this way, __name__ indirectly tells Flask where main.py lives on disk, not by providing the path itself, but by allowing Flask to look it up through Python’s import system.
- **CORS(app)** stands for ***Cross-Origin Resource Sharing*** - its browser security mechanism. It means attach CORS behavior to the Flask application instance (app). 
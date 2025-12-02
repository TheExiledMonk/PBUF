# 🎯 COSMOS2 REPORTING SYSTEM - COMPREHENSIVE REDESIGN PLAN

## 🚨 **CURRENT PROBLEMS IDENTIFIED**

### **❌ MAJOR ISSUES**
1. **Inconsistent Data Loading** - Sometimes tables load, sometimes they don't
2. **Tiny Tables** - Tables are unreadable with font-size: 0.75rem
3. **Missing Jackknife Data** - PBUF jackknife results are empty (0 items)
4. **No Modularity** - Everything is hardcoded in one giant class
5. **Inconsistent Output** - Different runs produce different report sections
6. **Zero Values** - Many tables contain 0s or empty data
7. **No Plugin Architecture** - Can't add/remove report sections easily
8. **Hard to Debug** - Monolithic code makes it impossible to fix individual issues

## 🏗️ **NEW ARCHITECTURE DESIGN**

### **📦 ENHANCED MODULAR PLUGIN SYSTEM**
```
cosmos2/
├── reporting/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── report_engine.py      # Main orchestrator
│   │   ├── data_loader.py         # Standardized data loading
│   │   ├── plugin_manager.py      # Plugin system manager
│   │   └── theme_manager.py       # Theme system manager
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── base_plugin.py         # Abstract base class
│   │   ├── data/                  # DATA PLUGINS (what to display)
│   │   │   ├── __init__.py
│   │   │   ├── model_comparison.py    # Model comparison section
│   │   │   ├── model_details.py       # Individual model sections
│   │   │   ├── jackknife_analysis.py  # Jackknife sections per model
│   │   │   ├── data_tables.py         # All data tables
│   │   │   ├── figures.py             # All figures/plots
│   │   │   ├── configuration.py       # Configuration display
│   │   │   ├── summary.py             # Executive summary
│   │   │   ├── conclusions.py         # Conclusions section
│   │   │   └── recommendations.py     # Recommendations section
│   │   └── output/                # OUTPUT PLUGINS (how to display)
│   │       ├── __init__.py
│   │       ├── base_output.py       # Base output plugin
│   │       ├── html_output.py       # HTML format output
│   │       ├── pdf_output.py        # PDF format output
│   │       ├── latex_output.py      # LaTeX format output
│   │       ├── markdown_output.py   # Markdown format output
│   │       ├── json_output.py       # JSON format output
│   │       └── csv_output.py        # CSV format output
│   ├── themes/
│   │   ├── __init__.py
│   │   ├── base_theme.py           # Base theme class
│   │   ├── html/                   # HTML THEMES (visual design)
│   │   │   ├── __init__.py
│   │   │   ├── professional.py      # Professional report theme
│   │   │   ├── academic.py          # Academic paper theme
│   │   │   ├── technical.py         # Technical report theme
│   │   │   ├── presentation.py      # Presentation slide theme
│   │   │   └── minimal.py           # Minimal clean theme
│   │   ├── templates/              # HTML TEMPLATES
│   │   │   ├── sections/           # Section templates
│   │   │   │   ├── header.html
│   │   │   │   ├── model_section.html
│   │   │   │   ├── table.html
│   │   │   │   ├── figure.html
│   │   │   │   └── comparison.html
│   │   │   ├── components/          # Component templates
│   │   │   │   ├── data_table.html
│   │   │   │   ├── parameter_grid.html
│   │   │   │   ├── stability_chart.html
│   │   │   │   └── navigation.html
│   │   │   └── layouts/            # Layout templates
│   │   │       ├── single_page.html
│   │   │       ├── multi_page.html
│   │   │       └── dashboard.html
│   │   └── css/                    # Theme CSS files
│   │       ├── professional.css
│   │       ├── academic.css
│   │       ├── technical.css
│   │       └── presentation.css
│   └── utils/
│       ├── __init__.py
│       ├── data_validation.py     # Data validation utilities
│       ├── table_formatter.py     # Table formatting utilities
│       ├── image_embedder.py      # Image embedding utilities
│       └── template_renderer.py   # Template rendering utilities
```

## 🔌 **ENHANCED PLUGIN ARCHITECTURE**

### **📊 Data Plugins (WHAT to display)**
```python
class DataPlugin(BasePlugin):
    """Base class for data plugins - defines WHAT data to display."""
    
    def get_data_type(self) -> str:
        """Return data type: 'model', 'comparison', 'summary', etc."""
        pass
    
    def get_section_type(self) -> str:
        """Return section type: 'individual', 'comparison', 'standalone'."""
        pass
    
    def get_target_models(self) -> List[str]:
        """Return which models this plugin applies to: ['lcdm', 'pbuf'] or ['all']."""
        pass
    
    def generate_data_structure(self) -> Dict[str, Any]:
        """Generate the data structure for this section."""
        pass
```

### **🖨️ Output Plugins (HOW to display)**
```python
class OutputPlugin(BasePlugin):
    """Base class for output plugins - defines HOW to display data."""
    
    def get_output_format(self) -> str:
        """Return format: 'html', 'pdf', 'latex', 'markdown', etc."""
        pass
    
    def render_data(self, data_structure: Dict[str, Any], theme: Theme) -> str:
        """Render data structure using theme."""
        pass
    
    def get_file_extension(self) -> str:
        """Return file extension: '.html', '.pdf', '.tex', etc."""
        pass
```

### **🎨 Theme System (VISUAL design)**
```python
class Theme:
    """Theme class for visual design."""
    
    def get_template(self, component_type: str) -> str:
        """Get HTML template for component type."""
        pass
    
    def get_css_styles(self) -> str:
        """Get CSS styles for this theme."""
        pass
    
    def get_layout_config(self) -> Dict[str, Any]:
        """Get layout configuration."""
        pass
```

## 🎯 **PLUGIN COMBINATION SYSTEM**

### **🔧 Plugin Manager Enhanced**
```python
class PluginManager:
    """Enhanced plugin manager with data/output/theme separation."""
    
    def __init__(self):
        self.data_plugins = {}      # WHAT to display
        self.output_plugins = {}     # HOW to display
        self.themes = {}             # VISUAL design
    
    def register_data_plugin(self, plugin: DataPlugin)
    def register_output_plugin(self, plugin: OutputPlugin)
    def register_theme(self, theme: Theme)
    
    def generate_report(self, 
                       data_plugins: List[str],     # Which data sections
                       output_format: str,          # How to render
                       theme_name: str) -> str:    # Visual theme
        """Generate report with specific plugins and theme."""
        pass
```

### **📋 Flexible Section Organization**
```python
# Data plugins can create:
# 1. Model-specific sections (under each model)
# 2. Comparison sections (between models)  
# 3. Standalone sections (conclusions, recommendations)

class ModelComparisonPlugin(DataPlugin):
    def get_section_type(self) -> str:
        return 'comparison'  # Goes between model sections
    
class JackknifeAnalysisPlugin(DataPlugin):
    def get_section_type(self) -> str:
        return 'individual'  # Goes under each model
    
class ConclusionsPlugin(DataPlugin):
    def get_section_type(self) -> str:
        return 'standalone'   # Separate section at end
```

## 🎨 **HTML THEME SYSTEM**

### **📝 Template-Based Design**
```html
<!-- themes/html/templates/sections/model_section.html -->
<section class="model-panel" data-model="{{ model_name }}">
    <h2>{{ model_name|upper }}</h2>
    <div class="model-meta">
        {% for meta_item in model_meta %}
        <span>{{ meta_item.label }}: {{ meta_item.value }}</span>
        {% endfor %}
    </div>
    
    {% include "components/parameter_grid.html" with parameters %}
    {% include "components/data_table.html" with chi2_breakdown %}
    
    {% if jackknife_data %}
    {% include "sections/jackknife_section.html" with jackknife_data %}
    {% endif %}
</section>
```

### **🎯 Component-Based Design**
```html
<!-- themes/html/templates/components/data_table.html -->
<div class="table-container">
    <table class="data-table {{ table_class }}">
        <thead>
            <tr>
                {% for header in table_headers %}
                <th>{{ header }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in table_rows %}
            <tr class="{{ row.class }}">
                {% for cell in row.cells %}
                <td>{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
```

### **🎨 Theme-Specific CSS**
```css
/* themes/html/css/professional.css */
.data-table {
    font-size: 0.9rem;           /* READABLE SIZE */
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}

.model-panel {
    background: white;
    border-radius: 0.8rem;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-left: 4px solid var(--model-color);
}

.lcdm { --model-color: #3b82f6; }
.pbuf { --model-color: #ef4444; }
```

## 🚀 **USAGE EXAMPLES**

### **📊 Basic Report Generation**
```python
# Generate professional HTML report with all sections
manager = PluginManager()
manager.generate_report(
    data_plugins=['model_comparison', 'model_details', 'jackknife_analysis', 'data_tables'],
    output_format='html',
    theme_name='professional'
)
```

### **🎨 Different Output Formats**
```python
# Same data, different formats
html_report = manager.generate_report(data_plugins, 'html', 'professional')
pdf_report = manager.generate_report(data_plugins, 'pdf', 'academic')
latex_report = manager.generate_report(data_plugins, 'latex', 'technical')
```

### **🔧 Custom Section Selection**
```python
# Only model comparison and conclusions
manager.generate_report(
    data_plugins=['model_comparison', 'conclusions'],
    output_format='html',
    theme_name='minimal'
)
```

### **🎯 Theme Customization**
```python
# Create custom theme by modifying templates
custom_theme = Theme(
    templates_dir='my_custom_templates/',
    css_file='my_custom.css'
)
manager.register_theme(custom_theme)
```

## 📋 **PLUGIN DEVELOPMENT GUIDE**

### **🔧 Creating New Data Plugins**
```python
class CustomAnalysisPlugin(DataPlugin):
    def get_data_type(self) -> str:
        return 'custom_analysis'
    
    def get_section_type(self) -> str:
        return 'individual'  # Under each model
    
    def get_target_models(self) -> List[str]:
        return ['lcdm', 'pbuf']
    
    def generate_data_structure(self) -> Dict[str, Any]:
        return {
            'custom_metrics': self.calculate_custom_metrics(),
            'visualizations': self.create_visualizations()
        }
```

### **🖨️ Creating New Output Formats**
```python
class ExcelOutputPlugin(OutputPlugin):
    def get_output_format(self) -> str:
        return 'excel'
    
    def render_data(self, data_structure: Dict[str, Any], theme: Theme) -> str:
        # Convert data to Excel format
        return self.generate_excel_workbook(data_structure)
```

### **🎨 Creating New Themes**
```html
<!-- themes/html/templates/sections/custom_model_section.html -->
<section class="custom-model-layout" data-model="{{ model_name }}">
    <div class="model-header">
        <h2>{{ model_name|upper }} Analysis</h2>
        <div class="model-status">{{ status }}</div>
    </div>
    
    <div class="two-column-layout">
        <div class="left-column">
            {% include "components/parameter_grid.html" %}
        </div>
        <div class="right-column">
            {% include "components/figure_gallery.html" %}
        </div>
    </div>
</section>
```

### **📋 Base Plugin Interface**
```python
class BasePlugin:
    """Base class for all report plugins."""
    
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
    
    def validate_data(self) -> bool:
        """Validate that required data is available."""
        pass
    
    def generate_html(self) -> str:
        """Generate HTML section."""
        pass
    
    def get_dependencies(self) -> List[str]:
        """Return list of required plugins."""
        pass
    
    def get_css_classes(self) -> List[str]:
        """Return CSS classes needed."""
        pass
```

### **🎯 Plugin Manager**
```python
class PluginManager:
    """Manages plugin loading, dependencies, and execution order."""
    
    def load_plugins(self, plugin_names: List[str])
    def resolve_dependencies(self)
    def validate_all_plugins(self) -> Dict[str, bool]
    def generate_report(self) -> str
```

## 📊 **DATA LAYER REDESIGN**

### **🔄 Standardized Data Loader**
```python
class DataLoader:
    """Standardized data loading with validation."""
    
    def load_model_data(self) -> Dict[str, Any]
    def load_jackknife_data(self) -> Dict[str, Any]
    def load_tables(self) -> Dict[str, pd.DataFrame]
    def load_figures(self) -> Dict[str, Path]
    def load_configuration(self) -> Dict[str, Any]
    def validate_data_integrity(self) -> Dict[str, bool]
    def get_data_summary(self) -> Dict[str, Any]
```

### **✅ Data Validation**
- **Check for empty tables** (all zeros, NaN values)
- **Validate jackknife data** (ensure both models have data)
- **Check figure files exist** and are readable
- **Validate configuration** completeness
- **Data type checking** (ensure correct data types)

## 🎨 **THEME SYSTEM**

### **🎯 Professional Theme**
- **Large, readable tables** (font-size: 0.9rem, proper spacing)
- **Consistent color scheme** (LCDM: blue, PBUF: red)
- **Professional typography** (Inter font family)
- **Responsive layout** (works on all screen sizes)
- **Print-friendly** CSS

### **📋 Table Styling**
```css
.data-table {
    font-size: 0.9rem;           /* READABLE SIZE */
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}
.data-table th,
.data-table td {
    padding: 0.75rem 1rem;      /* PROPER SPACING */
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}
```

## 🔍 **SPECIFIC PLUGINS TO CREATE**

### **1. Model Comparison Plugin**
- **Input**: Model summaries (χ², AIC, BIC, parameters)
- **Output**: Comparison table with neutral language
- **Features**: Δχ² calculation, significance levels
- **Validation**: Ensure both models have valid χ² values

### **2. Model Details Plugin**
- **Input**: Individual model data
- **Output**: Detailed sections for LCDM and PBUF
- **Features**: Parameters, χ² breakdown, derived quantities
- **Validation**: Check for missing/zero parameter values

### **3. Jackknife Analysis Plugin**
- **Input**: Jackknife level1/level2 results
- **Output**: Per-model jackknife sections
- **Features**: Parameter stability, strategy impact
- **Validation**: Ensure both models have jackknife data
- **Error Handling**: Show warnings when data is missing

### **4. Data Tables Plugin**
- **Input**: All CSV tables from tables/ directory
- **Output**: Formatted tables with proper sizing
- **Features**: Scrollable containers, sticky headers
- **Validation**: Remove empty tables, validate data types

### **5. Figures Plugin**
- **Input**: PNG figures from figures/ directory
- **Output**: Embedded images with captions
- **Features**: Base64 embedding, responsive sizing
- **Validation**: Check image integrity

### **6. Configuration Plugin**
- **Input**: Configuration JSON files
- **Output**: Collapsible configuration display
- **Features**: Syntax highlighting, search functionality
- **Validation**: Ensure config is complete

## 🐛 **DEBUGGING & LOGGING**

### **📝 Comprehensive Logging**
```python
class ReportLogger:
    """Detailed logging for report generation."""
    
    def log_data_loading(self, data_type: str, status: str, details: str)
    def log_plugin_validation(self, plugin_name: str, status: bool, issues: List[str])
    def log_data_issues(self, data_type: str, issues: List[str])
    def generate_debug_report(self) -> str
```

### **🔍 Data Validation Report**
- **Missing data identification** (what's missing and why)
- **Zero/empty data detection** (tables with all zeros)
- **Data type validation** (incorrect data types)
- **Jackknife data verification** (ensure both models work)

## 🚀 **IMPLEMENTATION PLAN**

### **PHASE 1: CORE INFRASTRUCTURE**
1. **Create reporting module structure**
2. **Implement BasePlugin class**
3. **Create PluginManager**
4. **Implement DataLoader with validation**
5. **Create HTML renderer**

### **PHASE 2: ESSENTIAL PLUGINS**
1. **Model Comparison Plugin** (highest priority)
2. **Data Tables Plugin** (fix table sizing)
3. **Model Details Plugin** (individual model sections)
4. **Professional Theme** (proper styling)

### **PHASE 3: ADVANCED PLUGINS**
1. **Jackknife Analysis Plugin** (fix jackknife issues)
2. **Figures Plugin** (embed all plots)
3. **Configuration Plugin** (show run config)
4. **Summary Plugin** (executive summary)

### **PHASE 4: INTEGRATION & TESTING**
1. **CLI integration** (replace current report command)
2. **Comprehensive testing** (with real science run data)
3. **Performance optimization** (large dataset handling)
4. **Documentation** (usage examples, plugin development guide)

## 📋 **QUALITY REQUIREMENTS**

### **✅ MUST-HAVE FEATURES**
1. **All data must be displayed** (no missing sections)
2. **Tables must be readable** (proper font size and spacing)
3. **Jackknife data for both models** (fix PBUF empty results)
4. **No zero/empty data** (validate and handle properly)
5. **Consistent output** (same sections every time)
6. **Professional appearance** (matching example report)

### **🔧 TECHNICAL REQUIREMENTS**
1. **Modular architecture** (easy to add/remove sections)
2. **Data validation** (catch and report issues)
3. **Error handling** (graceful degradation)
4. **Performance** (handle large datasets)
5. **Maintainability** (easy to debug and modify)

### **📊 USER EXPERIENCE**
1. **One command generation** (`python cosmos_cli.py report --run [DIR]`)
2. **Clear feedback** (what's working, what's missing)
3. **Professional output** (publication-ready reports)
4. **Comprehensive data** (everything we generate is shown)

## 🎯 **SUCCESS CRITERIA**

### **✅ FUNCTIONAL SUCCESS**
- [ ] All tables are readable and complete
- [ ] Jackknife analysis works for both models
- [ ] No zero/empty data in final report
- [ ] Consistent sections across all runs
- [ ] Professional styling matches example

### **🔧 TECHNICAL SUCCESS**
- [ ] Plugin system works (can add/remove sections)
- [ ] Data validation catches all issues
- [ ] CLI integration works seamlessly
- [ ] Performance acceptable for large datasets
- [ ] Code is maintainable and debuggable

### **👤 USER SUCCESS**
- [ ] Single command generates complete report
- [ ] Clear feedback about data status
- [ ] Professional output for presentations
- [ ] All generated data is visible
- [ ] Easy to understand and use

## 🚨 **IMMEDIATE NEXT STEPS**

1. **STOP** current report generator development
2. **CREATE** the modular reporting module structure
3. **IMPLEMENT** core infrastructure (BasePlugin, PluginManager, DataLoader)
4. **DEVELOP** essential plugins (Model Comparison, Data Tables, Model Details)
5. **FIX** jackknife data issues in the science runner
6. **TEST** with real science run data
7. **INTEGRATE** with CLI

## 📝 **NOTES**

- **This plan prioritizes getting ALL data displayed properly**
- **Modular architecture makes it easy to fix individual issues**
- **Data validation prevents silent failures**
- **Plugin system allows easy customization**
- **Professional theme ensures publication-ready output**

**The key insight: We need to build a robust foundation first, then add features systematically.**

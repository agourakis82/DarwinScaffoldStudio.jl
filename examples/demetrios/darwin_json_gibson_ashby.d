//! Darwin Scaffold Studio ↔ Demetrios JSON bridge example
//!
//! Contract:
//!   dc run examples/demetrios/darwin_json_gibson_ashby.d <input.json>
//!
//! Input JSON:
//!   { "porosity": 0.75, "E_solid": 1000.0 }
//!
//! Output JSON:
//!   { "porosity": 0.75, "E_solid": 1000.0, "E_scaffold": 62.5 }

module darwin_json_gibson_ashby

use io::env
use io::read_file
use json::{parse_json, JsonValue}

fn main() with IO {
    let args = env::args();
    if args.len() < 2 {
        var err = JsonValue::object();
        err.set("error".to_string(), JsonValue::string("missing_input_json_path".to_string()));
        println(err.to_json_string());
        return;
    }

    let path = args[1];
    let input_str = match read_file(path.as_str()) {
        Ok(s) => s,
        Err(e) => {
            var err = JsonValue::object();
            err.set("error".to_string(), JsonValue::string("read_file_failed".to_string()));
            err.set("path".to_string(), JsonValue::string(path.to_string()));
            println(err.to_json_string());
            return;
        }
    };

    let input = match parse_json(input_str.as_str()) {
        Ok(v) => v,
        Err(e) => {
            var err = JsonValue::object();
            err.set("error".to_string(), JsonValue::string("parse_json_failed".to_string()));
            println(err.to_json_string());
            return;
        }
    };

    let porosity = input.get("porosity").as_f64().unwrap_or(0.0);
    let e_solid = input.get("E_solid").as_f64().unwrap_or(0.0);
    let e_scaffold = e_solid * (1.0 - porosity) * (1.0 - porosity);

    var out = JsonValue::object();
    out.set("porosity".to_string(), JsonValue::number(porosity));
    out.set("E_solid".to_string(), JsonValue::number(e_solid));
    out.set("E_scaffold".to_string(), JsonValue::number(e_scaffold));
    println(out.to_json_string());
}

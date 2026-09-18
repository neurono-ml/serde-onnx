use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;
use syn::Lit;
use syn::Type;
use syn::parse_macro_input;
use syn::spanned::Spanned;

struct ContainerConfig {
    op: String,
    domain: String,
    input: String,
}

struct FieldConfig {
    ident: syn::Ident,
    attr: String,
    kind: AttrKind,
    optional: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AttrKind {
    Floats,
    Float,
    Ints,
    Int,
    Strings,
    String,
}

impl AttrKind {
    fn constructor(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Floats => quote!(floats),
            Self::Float => quote!(float),
            Self::Ints => quote!(ints),
            Self::Int => quote!(int),
            Self::Strings => quote!(strings),
            Self::String => quote!(string),
        }
    }

    fn from_ty_name(name: &str) -> Option<Self> {
        match name {
            "floats" => Some(Self::Floats),
            "float" => Some(Self::Float),
            "ints" => Some(Self::Ints),
            "int" => Some(Self::Int),
            "strings" => Some(Self::Strings),
            "string" => Some(Self::String),
            _ => None,
        }
    }
}

fn onnx_attrs(attrs: &[syn::Attribute]) -> syn::Result<Vec<(String, String, Span)>> {
    let mut out = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("onnx") {
            attr.meta
                .require_list()?
                .parse_nested_meta(|meta| {
                    let key = meta
                        .path
                        .get_ident()
                        .map(|i| i.to_string())
                        .ok_or_else(|| {
                            meta.error("expected key = \"value\" pairs inside #[onnx(...)]")
                        })?;
                    let span = meta.path.span();
                    let value: Lit = meta.value()?.parse()?;
                    match value {
                        Lit::Str(s) => out.push((key, s.value(), span)),
                        _ => return Err(meta.error("expected a string literal value")),
                    }
                    Ok(())
                })
                .map_err(|e| {
                    syn::Error::new(attr.span(), format!("invalid #[onnx(...)] annotation: {e}"))
                })?;
        }
    }
    Ok(out)
}

fn parse_container(input: &DeriveInput) -> syn::Result<ContainerConfig> {
    let mut op: Option<String> = None;
    let mut domain: Option<String> = None;
    let mut custom_input: Option<String> = None;
    for (key, value, span) in onnx_attrs(&input.attrs)? {
        match key.as_str() {
            "op" => op = Some(value),
            "domain" => domain = Some(value),
            "input" => custom_input = Some(value),
            _ => {
                return Err(syn::Error::new(
                    span,
                    format!("unknown container key {key:?}; expected one of op, domain, input"),
                ));
            }
        }
    }
    let op = op.ok_or_else(|| {
        syn::Error::new_spanned(
            &input.ident,
            "missing required #[onnx(op = \"...\")] on the struct",
        )
    })?;
    if op.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "op name must not be empty",
        ));
    }
    let domain = domain.ok_or_else(|| {
        syn::Error::new_spanned(
            &input.ident,
            "missing required #[onnx(domain = \"...\")] on the struct",
        )
    })?;
    if domain.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "domain must not be empty",
        ));
    }
    Ok(ContainerConfig {
        op,
        domain,
        input: custom_input.unwrap_or_else(|| "X".to_string()),
    })
}

fn strip_option(inner: &Type) -> Option<Type> {
    match inner {
        Type::Path(p) => {
            let seg = p.path.segments.last()?;
            if seg.ident != "Option" {
                return None;
            }
            match &seg.arguments {
                syn::PathArguments::AngleBracketed(args) => {
                    if args.args.len() != 1 {
                        return None;
                    }
                    match args.args.first()? {
                        syn::GenericArgument::Type(t) => Some(t.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn base_type_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(p) => {
            let seg = p.path.segments.last()?;
            let mut name = seg.ident.to_string();
            if name == "Vec" {
                match &seg.arguments {
                    syn::PathArguments::AngleBracketed(args) => {
                        if args.args.len() != 1 {
                            return None;
                        }
                        match args.args.first()? {
                            syn::GenericArgument::Type(inner) => {
                                let inner_name = base_type_name(inner)?;
                                name = format!("Vec<{inner_name}>");
                            }
                            _ => return None,
                        }
                    }
                    _ => return None,
                }
            }
            Some(name)
        }
        _ => None,
    }
}

fn infer_kind(ty: &Type) -> syn::Result<(AttrKind, bool)> {
    let (inner, optional) = match strip_option(ty) {
        Some(inner) => (inner, true),
        None => (ty.clone(), false),
    };
    let name = base_type_name(&inner).ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            "unsupported field type for #[onnx(attr = ...)]; expected Vec<f32>, f32, Vec<i64>, i64, Vec<String> or String, each optionally wrapped in Option",
        )
    })?;
    let kind = match name.as_str() {
        "Vec<f32>" => AttrKind::Floats,
        "f32" => AttrKind::Float,
        "Vec<i64>" => AttrKind::Ints,
        "i64" => AttrKind::Int,
        "Vec<String>" | "Vec<str>" => AttrKind::Strings,
        "String" | "str" => AttrKind::String,
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                format!(
                    "unsupported field type {name:?} for #[onnx(attr = ...)]; expected Vec<f32>, f32, Vec<i64>, i64, Vec<String> or String, each optionally wrapped in Option"
                ),
            ));
        }
    };
    Ok((kind, optional))
}

fn parse_field(field: &syn::Field) -> syn::Result<Option<FieldConfig>> {
    let ident = field.ident.clone().ok_or_else(|| {
        syn::Error::new_spanned(field, "OnnxExport only supports structs with named fields")
    })?;
    let mut attr_name: Option<String> = None;
    let mut ty_override: Option<String> = None;
    for (key, value, span) in onnx_attrs(&field.attrs)? {
        match key.as_str() {
            "attr" => attr_name = Some(value),
            "ty" => ty_override = Some(value),
            _ => {
                return Err(syn::Error::new(
                    span,
                    format!("unknown field key {key:?}; expected one of attr, ty"),
                ));
            }
        }
    }
    let attr = match attr_name {
        Some(a) => a,
        None => return Ok(None),
    };
    if attr.is_empty() {
        return Err(syn::Error::new_spanned(
            &field.ident,
            "attr name must not be empty",
        ));
    }
    let (mut kind, optional) = infer_kind(&field.ty)?;
    if let Some(ty_name) = ty_override {
        let forced = AttrKind::from_ty_name(&ty_name).ok_or_else(|| {
            syn::Error::new_spanned(
                &field.ty,
                format!(
                    "unknown ty {ty_name:?}; expected one of floats, float, ints, int, strings, string"
                ),
            )
        })?;
        if forced != kind {
            return Err(syn::Error::new_spanned(
                &field.ty,
                format!(
                    "ty {ty_name:?} does not match the Rust field type; remove ty or fix the field type"
                ),
            ));
        }
        kind = forced;
    }
    Ok(Some(FieldConfig {
        ident,
        attr,
        kind,
        optional,
    }))
}

#[proc_macro_derive(OnnxExport, attributes(onnx))]
pub fn derive_onnx_export(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let named = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) => n,
            _ => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "OnnxExport only supports structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "OnnxExport can only be derived for structs",
            ));
        }
    };
    let container = parse_container(input)?;
    let mut fields = Vec::new();
    for field in &named.named {
        if let Some(cfg) = parse_field(field)? {
            fields.push(cfg);
        }
    }
    if fields.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "OnnxExport requires at least one field annotated with #[onnx(attr = \"...\")]; fields without attr are not exported",
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for f in &fields {
        if !seen.insert(f.attr.clone()) {
            return Err(syn::Error::new_spanned(
                &f.ident,
                format!(
                    "duplicate attribute mapping {:?}; each attr name may appear once",
                    f.attr
                ),
            ));
        }
    }
    let struct_name = &input.ident;
    let op = &container.op;
    let domain = &container.domain;
    let input_name = &container.input;
    let mut pushers = Vec::new();
    for f in &fields {
        let ident = &f.ident;
        let attr = &f.attr;
        let ctor = f.kind.constructor();
        let optional = f.optional;
        let stmt = if optional {
            quote!(if let Some(value) = &self.#ident {
                attributes.push(::serde_onnx::ir::Attribute::#ctor(#attr, value.clone()));
            })
        } else {
            quote!(attributes.push(::serde_onnx::ir::Attribute::#ctor(#attr, self.#ident.clone()));)
        };
        pushers.push(stmt);
    }
    Ok(quote!(
        impl ::serde_onnx::export::ToOnnx for #struct_name {
            fn to_graph(
                &self,
                builder: &mut ::serde_onnx::export::GraphBuilder,
            ) -> ::core::result::Result<
                ::serde_onnx::export::ValueRef,
                ::serde_onnx::export::ExportError,
            > {
                let io_type = ::serde_onnx::ir::ValueType::tensor(
                    ::serde_onnx::ir::ElemType::Float,
                    ::core::option::Option::Some(vec![
                        ::serde_onnx::ir::Dim::Unknown,
                        ::serde_onnx::ir::Dim::Unknown,
                    ]),
                );
                let input = builder.input(#input_name, io_type.clone())?;
                let mut attributes: ::std::vec::Vec<::serde_onnx::ir::Attribute> =
                    ::std::vec::Vec::new();
                #(#pushers)*
                let mut outputs = builder.push_node(
                    #op,
                    #domain,
                    vec![input.name().to_string()],
                    attributes,
                    1,
                )?;
                let output = outputs.pop().expect("single output");
                builder.output(output.name().to_string(), io_type)?;
                Ok(output)
            }
        }
    ))
}

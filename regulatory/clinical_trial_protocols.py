import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

class ClinicalTrialProtocols:
    """
    Clinical trial protocols for FairTriEdge-FL medical AI system.
    Implements ICH-GCP guidelines and FDA guidance for AI/ML medical devices.
    """
    
    def __init__(self, study_title="FairTriEdge-FL Clinical Validation Study"):
        self.study_title = study_title
        self.protocol_version = "1.0"
        self.protocol_date = datetime.now()
        
        # Create clinical trial documentation directory
        Path("regulatory/clinical_trials").mkdir(parents=True, exist_ok=True)
    
    def generate_study_protocol(self):
        """Generate comprehensive clinical study protocol."""
        protocol = {
            "protocol_header": {
                "study_title": self.study_title,
                "protocol_number": "FTE-FL-001",
                "protocol_version": self.protocol_version,
                "protocol_date": self.protocol_date.isoformat(),
                "sponsor": "FairTriEdge-FL Research Consortium",
                "phase": "Validation Study (Non-interventional)",
                "study_type": "Prospective, Multi-center, Observational Study"
            },
            "study_objectives": {
                "primary_objective": {
                    "description": "To evaluate the clinical performance of FairTriEdge-FL AI system in emergency department triage",
                    "primary_endpoint": "Agreement between AI triage recommendation and expert clinical assessment (Cohen's kappa)",
                    "success_criteria": "Cohen's kappa ≥ 0.75 with 95% confidence interval lower bound > 0.70"
                },
                "secondary_objectives": [
                    {
                        "description": "Evaluate safety of AI-assisted triage",
                        "endpoint": "Rate of critical under-triage (missed high-acuity cases)",
                        "success_criteria": "Under-triage rate < 5% for critical cases"
                    },
                    {
                        "description": "Assess impact on emergency department efficiency",
                        "endpoint": "Time to treatment for high-acuity patients",
                        "success_criteria": "Non-inferiority to standard triage (margin: 10 minutes)"
                    },
                    {
                        "description": "Evaluate fairness across patient populations",
                        "endpoint": "Performance consistency across demographic groups",
                        "success_criteria": "No significant performance differences by age, gender, or ethnicity"
                    },
                    {
                        "description": "Assess healthcare provider acceptance",
                        "endpoint": "Provider satisfaction and trust scores",
                        "success_criteria": "Mean satisfaction score ≥ 4.0 on 5-point scale"
                    }
                ]
            },
            "study_design": {
                "design_type": "Prospective, multi-center, non-randomized, observational study",
                "study_population": "Patients presenting to emergency departments",
                "study_duration": {
                    "enrollment_period": "12 months",
                    "follow_up_period": "30 days post-ED visit",
                    "total_study_duration": "15 months"
                },
                "sample_size": {
                    "target_enrollment": 10000,
                    "power_calculation": {
                        "primary_endpoint": "Cohen's kappa agreement",
                        "expected_kappa": 0.80,
                        "null_hypothesis_kappa": 0.70,
                        "alpha": 0.05,
                        "power": 0.80,
                        "calculated_sample_size": 8500,
                        "inflation_for_dropouts": "15%",
                        "final_sample_size": 10000
                    }
                }
            },
            "study_population": {
                "inclusion_criteria": [
                    "Patients ≥ 1 year of age presenting to emergency department",
                    "Complete vital signs available at triage",
                    "Informed consent obtained (or waiver approved by IRB)",
                    "English or Spanish speaking (or interpreter available)"
                ],
                "exclusion_criteria": [
                    "Patients requiring immediate resuscitation (ESI Level 1)",
                    "Patients with incomplete triage data",
                    "Patients who decline participation",
                    "Prisoners or other vulnerable populations (unless specifically approved)"
                ],
                "withdrawal_criteria": [
                    "Patient withdrawal of consent",
                    "Investigator decision for safety reasons",
                    "Protocol violation that compromises data integrity"
                ]
            },
            "study_procedures": {
                "screening_and_enrollment": {
                    "screening_process": "Automated screening of all ED presentations",
                    "consent_process": "Informed consent or IRB-approved waiver",
                    "randomization": "Not applicable (observational study)"
                },
                "data_collection": {
                    "baseline_data": [
                        "Demographics (age, gender, race/ethnicity)",
                        "Presenting complaint and chief complaint",
                        "Vital signs (blood pressure, heart rate, temperature, respiratory rate, oxygen saturation)",
                        "Pain score and mental status",
                        "Medical history and current medications",
                        "Arrival mode (ambulance, walk-in, etc.)"
                    ],
                    "ai_system_data": [
                        "AI triage recommendation (Green/Yellow/Red)",
                        "Confidence score for recommendation",
                        "Key features influencing decision",
                        "Processing time for recommendation"
                    ],
                    "clinical_data": [
                        "Clinical triage decision by nurse",
                        "Physician assessment and diagnosis",
                        "Diagnostic tests ordered and results",
                        "Treatment provided",
                        "Disposition (discharge, admission, transfer)"
                    ],
                    "outcome_data": [
                        "Length of stay in emergency department",
                        "Time to physician evaluation",
                        "Time to treatment initiation",
                        "Patient satisfaction scores",
                        "Adverse events during ED stay",
                        "30-day outcomes (readmission, mortality)"
                    ]
                }
            },
            "statistical_analysis_plan": {
                "primary_analysis": {
                    "endpoint": "Agreement between AI and clinical triage",
                    "statistical_method": "Cohen's kappa with 95% confidence interval",
                    "analysis_population": "Per-protocol population",
                    "handling_of_missing_data": "Complete case analysis for primary endpoint"
                },
                "secondary_analyses": [
                    {
                        "endpoint": "Sensitivity and specificity for high-acuity cases",
                        "method": "Receiver operating characteristic (ROC) analysis"
                    },
                    {
                        "endpoint": "Time to treatment analysis",
                        "method": "Kaplan-Meier survival analysis and Cox regression"
                    },
                    {
                        "endpoint": "Subgroup analyses",
                        "method": "Stratified analysis by age group, gender, and presenting complaint"
                    }
                ],
                "interim_analyses": {
                    "planned_interim_analyses": 2,
                    "timing": "After 25% and 50% of target enrollment",
                    "stopping_rules": "Futility or safety concerns based on DSMB recommendations"
                }
            }
        }
        
        # Save study protocol
        with open("regulatory/clinical_trials/study_protocol.json", "w") as f:
            json.dump(protocol, f, indent=2)
        
        return protocol
    
    def generate_informed_consent_form(self):
        """Generate informed consent form for study participants."""
        consent_form = {
            "document_title": "Informed Consent for Research Participation",
            "study_title": self.study_title,
            "protocol_number": "FTE-FL-001",
            "version_date": self.protocol_date.isoformat(),
            "sections": {
                "introduction": {
                    "text": "You are being asked to participate in a research study to evaluate an artificial intelligence (AI) system that helps healthcare providers make triage decisions in the emergency department. This consent form explains the study and your rights as a research participant."
                },
                "purpose_of_study": {
                    "text": "The purpose of this study is to test whether an AI computer system can help emergency department staff make better and faster decisions about how urgently patients need to be seen by a doctor. The AI system will analyze information about your symptoms and vital signs to suggest whether you should be seen immediately, soon, or can wait longer."
                },
                "study_procedures": {
                    "text": "If you agree to participate:\n• The AI system will analyze your medical information when you arrive at the emergency department\n• Your regular medical care will not change - doctors and nurses will still make all decisions about your care\n• We will collect information about your emergency department visit and follow up 30 days later\n• Your participation will not delay or change your medical treatment"
                },
                "risks_and_discomforts": {
                    "text": "The risks of participating in this study are minimal:\n• There is a small risk that the AI system could make an incorrect recommendation, but your doctors and nurses will always make the final decisions about your care\n• There is a risk of loss of privacy, but we have strong protections in place to keep your information secure\n• You may receive follow-up calls or surveys about your experience"
                },
                "benefits": {
                    "text": "You may not receive any direct benefit from participating in this study. However, the information learned may help improve emergency department care for future patients by making triage decisions more accurate and consistent."
                },
                "privacy_and_confidentiality": {
                    "text": "Your privacy is very important to us:\n• Your medical information will be kept confidential according to HIPAA regulations\n• The AI system uses advanced privacy protection techniques that prevent identification of individual patients\n• Only authorized research staff will have access to your identifiable information\n• Research results will be published without any information that could identify you"
                },
                "voluntary_participation": {
                    "text": "Your participation in this research is entirely voluntary:\n• You can choose not to participate without affecting your medical care\n• You can withdraw from the study at any time\n• Your decision will not affect your relationship with your healthcare providers or this hospital"
                },
                "contact_information": {
                    "principal_investigator": {
                        "name": "Dr. [Principal Investigator Name]",
                        "phone": "[Phone Number]",
                        "email": "[Email Address]"
                    },
                    "irb_contact": {
                        "name": "Institutional Review Board",
                        "phone": "[IRB Phone Number]",
                        "email": "[IRB Email Address]"
                    }
                },
                "consent_statement": {
                    "text": "I have read this consent form and have had the opportunity to ask questions. I understand that my participation is voluntary and that I can withdraw at any time. I agree to participate in this research study."
                }
            },
            "signature_section": {
                "participant_signature": {
                    "signature_line": "Participant Signature",
                    "date_line": "Date",
                    "print_name_line": "Print Name"
                },
                "witness_signature": {
                    "signature_line": "Witness Signature (if required)",
                    "date_line": "Date",
                    "print_name_line": "Print Name"
                },
                "investigator_signature": {
                    "signature_line": "Person Obtaining Consent",
                    "date_line": "Date",
                    "print_name_line": "Print Name and Title"
                }
            }
        }
        
        # Save informed consent form
        with open("regulatory/clinical_trials/informed_consent_form.json", "w") as f:
            json.dump(consent_form, f, indent=2)
        
        return consent_form
    
    def generate_case_report_form(self):
        """Generate case report form (CRF) for data collection."""
        crf = {
            "form_title": "FairTriEdge-FL Clinical Study Case Report Form",
            "version": "1.0",
            "date": self.protocol_date.isoformat(),
            "sections": {
                "patient_identification": {
                    "study_id": {"type": "text", "required": True, "description": "Unique study identifier"},
                    "site_id": {"type": "text", "required": True, "description": "Study site identifier"},
                    "enrollment_date": {"type": "date", "required": True, "description": "Date of enrollment"},
                    "consent_obtained": {"type": "boolean", "required": True, "description": "Informed consent obtained"}
                },
                "demographics": {
                    "age": {"type": "integer", "required": True, "range": [1, 120], "units": "years"},
                    "gender": {"type": "select", "required": True, "options": ["Male", "Female", "Other", "Prefer not to answer"]},
                    "race": {"type": "select", "required": False, "options": ["White", "Black/African American", "Asian", "Native American", "Pacific Islander", "Other", "Unknown"]},
                    "ethnicity": {"type": "select", "required": False, "options": ["Hispanic/Latino", "Not Hispanic/Latino", "Unknown"]},
                    "primary_language": {"type": "text", "required": False, "description": "Primary language spoken"}
                },
                "presentation_data": {
                    "arrival_time": {"type": "datetime", "required": True, "description": "Time of arrival to ED"},
                    "arrival_mode": {"type": "select", "required": True, "options": ["Walk-in", "Private vehicle", "Ambulance", "Helicopter", "Other"]},
                    "chief_complaint": {"type": "text", "required": True, "description": "Primary reason for ED visit"},
                    "symptom_duration": {"type": "text", "required": False, "description": "Duration of symptoms"},
                    "pain_score": {"type": "integer", "required": False, "range": [0, 10], "description": "Pain score (0-10 scale)"}
                },
                "vital_signs": {
                    "systolic_bp": {"type": "integer", "required": True, "range": [50, 300], "units": "mmHg"},
                    "diastolic_bp": {"type": "integer", "required": True, "range": [30, 200], "units": "mmHg"},
                    "heart_rate": {"type": "integer", "required": True, "range": [30, 250], "units": "bpm"},
                    "respiratory_rate": {"type": "integer", "required": True, "range": [5, 60], "units": "breaths/min"},
                    "temperature": {"type": "decimal", "required": True, "range": [32.0, 45.0], "units": "°C"},
                    "oxygen_saturation": {"type": "integer", "required": True, "range": [50, 100], "units": "%"},
                    "weight": {"type": "decimal", "required": False, "range": [1.0, 300.0], "units": "kg"}
                },
                "medical_history": {
                    "allergies": {"type": "text", "required": False, "description": "Known allergies"},
                    "medications": {"type": "text", "required": False, "description": "Current medications"},
                    "medical_conditions": {"type": "text", "required": False, "description": "Significant medical history"},
                    "previous_surgeries": {"type": "text", "required": False, "description": "Previous surgical procedures"}
                },
                "ai_system_output": {
                    "ai_triage_recommendation": {"type": "select", "required": True, "options": ["Green", "Yellow", "Red"]},
                    "ai_confidence_score": {"type": "decimal", "required": True, "range": [0.0, 1.0], "description": "AI confidence (0-1)"},
                    "ai_processing_time": {"type": "decimal", "required": True, "units": "seconds", "description": "Time for AI to generate recommendation"},
                    "ai_key_features": {"type": "text", "required": False, "description": "Key features influencing AI decision"}
                },
                "clinical_assessment": {
                    "nurse_triage_level": {"type": "select", "required": True, "options": ["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"]},
                    "nurse_triage_time": {"type": "datetime", "required": True, "description": "Time of nurse triage completion"},
                    "physician_assessment_time": {"type": "datetime", "required": False, "description": "Time of initial physician assessment"},
                    "primary_diagnosis": {"type": "text", "required": False, "description": "Primary diagnosis (ICD-10)"},
                    "secondary_diagnoses": {"type": "text", "required": False, "description": "Secondary diagnoses"}
                },
                "treatment_and_disposition": {
                    "diagnostic_tests": {"type": "text", "required": False, "description": "Diagnostic tests ordered"},
                    "treatments_provided": {"type": "text", "required": False, "description": "Treatments and interventions"},
                    "disposition": {"type": "select", "required": True, "options": ["Discharge", "Admission", "Transfer", "Left AMA", "Deceased"]},
                    "ed_length_of_stay": {"type": "decimal", "required": True, "units": "hours", "description": "Total ED length of stay"},
                    "time_to_treatment": {"type": "decimal", "required": False, "units": "minutes", "description": "Time from arrival to first treatment"}
                },
                "safety_outcomes": {
                    "adverse_events": {"type": "text", "required": False, "description": "Any adverse events during ED stay"},
                    "medication_errors": {"type": "boolean", "required": True, "description": "Any medication errors"},
                    "diagnostic_errors": {"type": "text", "required": False, "description": "Any diagnostic errors or delays"},
                    "patient_complaints": {"type": "text", "required": False, "description": "Patient or family complaints"}
                },
                "follow_up_data": {
                    "30_day_readmission": {"type": "boolean", "required": True, "description": "Readmission within 30 days"},
                    "30_day_mortality": {"type": "boolean", "required": True, "description": "Death within 30 days"},
                    "patient_satisfaction": {"type": "integer", "required": False, "range": [1, 5], "description": "Patient satisfaction score (1-5)"},
                    "provider_satisfaction": {"type": "integer", "required": False, "range": [1, 5], "description": "Provider satisfaction with AI system (1-5)"}
                }
            },
            "data_validation_rules": {
                "required_field_checks": "All required fields must be completed",
                "range_checks": "Numeric values must be within specified ranges",
                "consistency_checks": "Cross-field validation for logical consistency",
                "completeness_checks": "Minimum data requirements for analysis"
            }
        }
        
        # Save case report form
        with open("regulatory/clinical_trials/case_report_form.json", "w") as f:
            json.dump(crf, f, indent=2)
        
        return crf
    
    def generate_data_management_plan(self):
        """Generate data management plan for clinical trial."""
        data_management_plan = {
            "plan_title": "Data Management Plan - FairTriEdge-FL Clinical Study",
            "version": "1.0",
            "date": self.protocol_date.isoformat(),
            "data_management_overview": {
                "data_types": [
                    "Clinical data from electronic health records",
                    "AI system outputs and performance metrics",
                    "Patient-reported outcome measures",
                    "Provider satisfaction surveys"
                ],
                "data_volume": "Estimated 10,000 patient records with ~200 data points each",
                "data_sources": [
                    "Hospital electronic health record systems",
                    "FairTriEdge-FL AI system logs",
                    "Patient surveys and questionnaires",
                    "Provider feedback forms"
                ]
            },
            "data_collection": {
                "collection_methods": {
                    "electronic_data_capture": "REDCap or similar HIPAA-compliant system",
                    "automated_data_extraction": "Direct integration with EHR systems where possible",
                    "manual_data_entry": "For data not available electronically",
                    "quality_control": "Double data entry for critical variables"
                },
                "data_collection_timeline": {
                    "baseline_data": "At time of ED presentation",
                    "intervention_data": "During ED stay",
                    "outcome_data": "At discharge and 30-day follow-up"
                }
            },
            "data_quality_assurance": {
                "data_validation": {
                    "real_time_validation": "Range checks and consistency rules in EDC system",
                    "periodic_data_review": "Monthly data quality reports",
                    "source_data_verification": "10% random sample verification against source documents"
                },
                "missing_data_handling": {
                    "prevention": "Required field validation in EDC system",
                    "monitoring": "Regular missing data reports",
                    "imputation": "Multiple imputation for key variables if appropriate"
                },
                "data_cleaning": {
                    "outlier_detection": "Statistical methods to identify outliers",
                    "consistency_checks": "Cross-variable validation rules",
                    "query_management": "Formal query process for data discrepancies"
                }
            },
            "data_security_and_privacy": {
                "access_controls": {
                    "user_authentication": "Multi-factor authentication required",
                    "role_based_access": "Access limited based on job function",
                    "audit_trails": "All data access logged and monitored"
                },
                "data_encryption": {
                    "data_at_rest": "AES-256 encryption for stored data",
                    "data_in_transit": "TLS 1.3 for all data transmissions",
                    "backup_encryption": "Encrypted backups with secure key management"
                },
                "privacy_protection": {
                    "de_identification": "Direct identifiers removed from analysis datasets",
                    "limited_datasets": "Minimum necessary data for research purposes",
                    "data_use_agreements": "Formal agreements with all participating sites"
                }
            },
            "data_storage_and_retention": {
                "primary_storage": {
                    "location": "HIPAA-compliant cloud infrastructure",
                    "backup_strategy": "Daily automated backups with geographic redundancy",
                    "disaster_recovery": "Comprehensive disaster recovery plan"
                },
                "retention_schedule": {
                    "active_study_period": "All data retained during study",
                    "post_study_retention": "25 years per FDA requirements",
                    "destruction_schedule": "Secure destruction after retention period"
                }
            },
            "data_sharing_and_dissemination": {
                "internal_sharing": {
                    "study_team_access": "Role-based access for study personnel",
                    "sponsor_access": "Aggregate data and reports",
                    "dsmb_access": "Safety data for Data Safety Monitoring Board"
                },
                "external_sharing": {
                    "regulatory_submissions": "De-identified data for FDA submission",
                    "publication_data": "Summary statistics for peer-reviewed publications",
                    "data_repository": "Consider sharing de-identified data in public repository"
                }
            }
        }
        
        # Save data management plan
        with open("regulatory/clinical_trials/data_management_plan.json", "w") as f:
            json.dump(data_management_plan, f, indent=2)
        
        return data_management_plan
    
    def generate_monitoring_plan(self):
        """Generate clinical trial monitoring plan."""
        monitoring_plan = {
            "plan_title": "Clinical Trial Monitoring Plan - FairTriEdge-FL Study",
            "version": "1.0",
            "date": self.protocol_date.isoformat(),
            "monitoring_overview": {
                "monitoring_approach": "Risk-based monitoring with central and on-site components",
                "monitoring_frequency": "Monthly central monitoring, quarterly on-site visits",
                "monitoring_scope": "All critical data and safety parameters"
            },
            "central_monitoring": {
                "data_review": {
                    "frequency": "Weekly automated reports, monthly detailed review",
                    "parameters_monitored": [
                        "Enrollment rates and demographics",
                        "Protocol deviations and violations",
                        "Data quality metrics",
                        "Safety signals and adverse events",
                        "Primary endpoint data completeness"
                    ]
                },
                "statistical_monitoring": {
                    "enrollment_tracking": "Monitor against target enrollment timeline",
                    "data_quality_metrics": "Missing data rates, outlier detection",
                    "safety_monitoring": "Adverse event rates and patterns",
                    "efficacy_monitoring": "Interim efficacy analyses per protocol"
                }
            },
            "on_site_monitoring": {
                "visit_schedule": {
                    "initiation_visit": "Before first patient enrollment",
                    "routine_visits": "Quarterly or after every 50 patients",
                    "close_out_visit": "After last patient last visit"
                },
                "monitoring_activities": [
                    "Source data verification (10% sample)",
                    "Informed consent review",
                    "Investigator file review",
                    "Protocol compliance assessment",
                    "Safety data review"
                ]
            },
            "data_safety_monitoring_board": {
                "composition": [
                    "Independent emergency medicine physician (Chair)",
                    "Biostatistician with clinical trials experience",
                    "AI/ML expert with healthcare experience",
                    "Patient safety expert",
                    "Bioethicist"
                ],
                "responsibilities": [
                    "Review safety data and adverse events",
                    "Monitor study conduct and data quality",
                    "Make recommendations on study continuation",
                    "Review interim efficacy analyses"
                ],
                "meeting_schedule": "Quarterly or as needed for safety concerns"
            }
        }
        
        # Save monitoring plan
        with open("regulatory/clinical_trials/monitoring_plan.json", "w") as f:
            json.dump(monitoring_plan, f, indent=2)
        
        return monitoring_plan
    
    def generate_complete_clinical_trial_package(self):
        """Generate complete clinical trial documentation package."""
        print("🏥 Generating Clinical Trial Protocols for FairTriEdge-FL")
        
        # Generate all protocol documents
        study_protocol = self.generate_study_protocol()
        informed_consent = self.generate_informed_consent_form()
        case_report_form = self.generate_case_report_form()
        data_management_plan = self.generate_data_management_plan()
        monitoring_plan = self.generate_monitoring_plan()
        
        # Generate summary package
        trial_package = {
            "package_title": "FairTriEdge-FL Clinical Trial Documentation Package",
            "version": "1.0",
            "date": self.protocol_date.isoformat(),
            "documents_included": {
                "study_protocol": "regulatory/clinical_trials/study_protocol.json",
                "informed_consent_form": "regulatory/clinical_trials/informed_consent_form.json",
                "case_report_form": "regulatory/clinical_trials/case_report_form.json",
                "data_management_plan": "regulatory/clinical_trials/data_management_plan.json",
                "monitoring_plan": "regulatory/clinical_trials/monitoring_plan.json"
            },
            "regulatory_requirements": {
                "ich_gcp_compliance": "All documents follow ICH-GCP guidelines",
                "fda_requirements": "Compliant with FDA guidance for AI/ML devices",
                "irb_submission": "Ready for Institutional Review Board submission",
                "regulatory_submission": "Suitable for FDA 510(k) or IDE submission"
            },
            "implementation_timeline": {
                "protocol_finalization": "Month 1",
                "irb_submission": "Month 2",
                "site_initiation": "Month 3-4",
                "patient_enrollment": "Month 5-16",
                "data_analysis": "Month 17-18",
                "final_report": "Month 19"
            },
            "resource_requirements": {
                "study_sites": "5-10 emergency departments",
                "study_personnel": "Principal investigator, study coordinators, data managers",
                "technology_requirements": "EDC system, AI system integration",
                "estimated_budget": "Contact for detailed budget estimate"
            }
        }
        
        # Save complete package summary
        with open("regulatory/clinical_trials/clinical_trial_package.json", "w") as f:
            json.dump(trial_package, f, indent=2)
        
        print("✅ Clinical Trial Protocols Complete")
        print(f"📁 Documentation saved to: regulatory/clinical_trials/")
        print(f"📋 Generated {len(trial_package['documents_included'])} protocol documents")
        
        return trial_package

def run_clinical_trial_protocols():
    """Run complete clinical trial protocol generation."""
    protocols = ClinicalTrialProtocols()
    return protocols.generate_complete_clinical_trial_package()

if __name__ == "__main__":
    run_clinical_trial_protocols()
package cse6250.fileio

import java.sql.Timestamp


case class Patient(
                    row_id: Int,
                    subject_id: Int,
                    gender: Option[String],
                    dob: Option[Timestamp],
                    dod: Option[Timestamp],
                    dod_hosp: Option[Timestamp],
                    dod_ssn: Option[Timestamp],
                    expire_flag: Option[String]
                  )

case class Admission(
                      row_id: Int,
                      subject_id: Int,
                      hadm_id: Int,
                      admit_time: Option[Timestamp],
                      discharge_time: Option[Timestamp],
                      death_time: Option[Timestamp],
                      admit_type: Option[String],
                      admit_loc: Option[String],
                      discharge_loc: Option[String],
                      insurance: Option[String],
                      language: Option[String],
                      religion: Option[String],
                      marital_status: Option[String],
                      ethnicity: Option[String],
                      ed_register_time: Option[Timestamp],
                      ed_checkout_time: Option[Timestamp],
                      diagnosis: Option[String],
                      hosp_exp_flag: Option[Int],
                      has_chartevents_data: Option[Int]
                    )

case class Note(
                 row_id: Int,
                 subject_id: Int,
                 hadm_id: Float,
                 chart_date: Option[Timestamp],
                 chart_time: Option[Timestamp],
                 store_time: Option[Timestamp],
                 category: Option[String],
                 description: Option[String],
                 cgid: Option[Int],
                 iserror: Option[Int],
                 text: Option[String]
               )

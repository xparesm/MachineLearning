﻿using Microsoft.ML.Data;

namespace MLMulticlassExperiment
{
    public class ModelInput
    {

        [ColumnName("delta"), LoadColumn(0)]
        public float Delta { get; set; }

        [ColumnName("b_anteriors"), LoadColumn(1)]
        public string BarresAnteriors { get; set; }

        [ColumnName("sitaucioVWAP"), LoadColumn(2)]
        public string SituacioVWAP { get; set; }

        [ColumnName("b_futures"), LoadColumn(3)]
        public string BarresFutures { get; set; }


        /*[ColumnName("id"), LoadColumn(0)]
        public float Id { get; set; }


        [ColumnName("hotel"), LoadColumn(1)]
        public string Hotel { get; set; }


        //[ColumnName("is_canceled"), LoadColumn(2)]
        //public float Is_canceled { get; set; }


        [ColumnName("lead_time"), LoadColumn(3)]
        public float Lead_time { get; set; }


        //[ColumnName("arrival_date_year"), LoadColumn(4)]
        //public float Arrival_date_year { get; set; }


        [ColumnName("arrival_date_month"), LoadColumn(5)]
        public string Arrival_date_month { get; set; }


        [ColumnName("arrival_date_week_number"), LoadColumn(6)]
        public float Arrival_date_week_number { get; set; }


        [ColumnName("arrival_date_day_of_month"), LoadColumn(7)]
        public float Arrival_date_day_of_month { get; set; }


        [ColumnName("stays_in_weekend_nights"), LoadColumn(8)]
        public float Stays_in_weekend_nights { get; set; }


        [ColumnName("stays_in_week_nights"), LoadColumn(9)]
        public float Stays_in_week_nights { get; set; }


        [ColumnName("adults"), LoadColumn(10)]
        public float Adults { get; set; }


        [ColumnName("children"), LoadColumn(11)]
        public float Children { get; set; }


        [ColumnName("babies"), LoadColumn(12)]
        public float Babies { get; set; }


        [ColumnName("meal"), LoadColumn(13)]
        public string Meal { get; set; }


        [ColumnName("country"), LoadColumn(14)]
        public string Country { get; set; }


        [ColumnName("market_segment"), LoadColumn(15)]
        public string Market_segment { get; set; }


        [ColumnName("distribution_channel"), LoadColumn(16)]
        public string Distribution_channel { get; set; }


        [ColumnName("is_repeated_guest"), LoadColumn(17)]
        public float Is_repeated_guest { get; set; }


        [ColumnName("previous_cancellations"), LoadColumn(18)]
        public float Previous_cancellations { get; set; }


        [ColumnName("previous_bookings_not_canceled"), LoadColumn(19)]
        public float Previous_bookings_not_canceled { get; set; }


        [ColumnName("reserved_room_type"), LoadColumn(20)]
        public string Reserved_room_type { get; set; }


        [ColumnName("assigned_room_type"), LoadColumn(21)]
        public string Assigned_room_type { get; set; }


        [ColumnName("booking_changes"), LoadColumn(22)]
        public float Booking_changes { get; set; }


        [ColumnName("deposit_type"), LoadColumn(23)]
        public string Deposit_type { get; set; }


        [ColumnName("agent"), LoadColumn(24)]
        public string Agent { get; set; }*/


        //[ColumnName("company"), LoadColumn(25)]
        //public string Company { get; set; }


        /*[ColumnName("days_in_waiting_list"), LoadColumn(26)]
        public float Days_in_waiting_list { get; set; }


        [ColumnName("customer_type"), LoadColumn(27)]
        public string Customer_type { get; set; }


        [ColumnName("adr"), LoadColumn(28)]
        public float Adr { get; set; }


        [ColumnName("required_car_parking_spaces"), LoadColumn(29)]
        public float Required_car_parking_spaces { get; set; }


        [ColumnName("total_of_special_requests"), LoadColumn(30)]
        public float Total_of_special_requests { get; set; }


        [ColumnName("reservation_status"), LoadColumn(31)]
        public string Reservation_status { get; set; }*/


        //[ColumnName("reservation_status_date"), LoadColumn(32)]
        //public string Reservation_status_date { get; set; }

    }
}
